import numpy as np
import pickle
import time
import jax.numpy as jnp
import optax
from jax import random, vmap, hessian, jacfwd, jit, value_and_grad, grad, lax
from tqdm import tqdm
from jax_md import space
import potentials
import utils
from jax_transformations3d import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
import pdb
import itertools
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
SEED = 42
key = random.PRNGKey(SEED)

# Targets
targets = [
    {"structure": [1, 0, 2, 3, 0, 4, 5, 0, 6], "desired_yield": 0.3},
]

target_shapes = [t["structure"] for t in targets]

# Set to True if optimizing over strengths of specific patch pairs
use_custom_pairs = False
custom_pairs = None

# Load species
def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('species_combinations.pkl')

mon_species = data['mon_pc_species']
dim_species = data['dimer_pc_species']
tri_species = data['trimer_pc_species']

n_m = mon_species.shape[0]
n_d = dim_species.shape[0]
n_t = tri_species.shape[0]

tot_num_structures = n_m + n_d + n_t

# Determine number of monomers
num_monomers = len([key for key in data.keys() if key.endswith('_mon_counts')])

# Helper functions
def indx_of_target(target):
    target = jnp.array(target)
    target_reversed = target[::-1]
    
    if target.shape[0] == 1 * 3:
        for i in range(n_m):
            if jnp.array_equal(mon_species[i], target) or jnp.array_equal(mon_species[i], target_reversed):
                return i
                
    elif target.shape[0] == 2 * 3:
        for i in range(n_d):
            if jnp.array_equal(dim_species[i], target) or jnp.array_equal(dim_species[i], target_reversed):
                return i + n_m
                
    elif target.shape[0] == 3 * 3:
        for i in range(n_t):
            if jnp.array_equal(tri_species[i], target) or jnp.array_equal(tri_species[i], target_reversed):
                return i + n_m + n_d

inx_targets = [indx_of_target(t["structure"]) for t in targets]
desired_yields = [t["desired_yield"] for t in targets]

euler_scheme = "sxyz"

# Constants
V = 1250.0
kT = 1.0
n = 3  # number of monomers

# Shape and energy helper functions
a = 1.0  # distance of the center of the spheres from the BB COM
b = 0.3  # distance of the center of the patches from the BB COM
separation = 2.0
noise = 1e-14
vertex_radius = a
patch_radius = 0.2 * a
small_value = 1e-12
vertex_species = 0
n_patches = n * 2  # 2 species of patches per monomer type
n_species = n_patches + 1  # plus the common vertex species 0

n_morse_vals = n_patches * (n_patches - 1) // 2 + n_patches  # all possible pair permutations plus same patch attraction (i,i)
patchy_vals = jnp.full(n_morse_vals, 4.0)  # FIXME for optimization over specific attraction strengths

# Generate initial concentrations dynamically based on the number of monomers
initial_concentrations = jnp.full(num_monomers, 0.15)
concs = initial_concentrations

# Combine initial parameters
init_params = jnp.concatenate([patchy_vals, concs])

shape = jnp.array([
    [-a, 0., b],  # first patch
    [-a, b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],  # second patch
    [-a, -b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],
    [0., 0., a],
    [0., a * jnp.cos(jnp.pi / 6.), -a * jnp.sin(jnp.pi / 6.)],  # second sphere
    [0., -a * jnp.cos(jnp.pi / 6.), -a * jnp.sin(jnp.pi / 6.)],  # third sphere
    [a, 0., b],  # first patch
    [a, b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],  # second patch
    [a, -b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)]  # third patch
])

mon_shape = jnp.array([shape])
dimer_shape = jnp.array([shape, shape])
trimer_shape = jnp.array([shape, shape, shape])

mon_rb = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
dimer_rb = jnp.array([-separation / 2.0, noise, 0, 0, 0, 0, separation / 2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)
trimer_rb = jnp.array([-separation, noise, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, separation, noise, 0, 0, 0, 0], dtype=jnp.float64)

rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)
rep_A_table = jnp.full((n_species, n_species), small_value).at[vertex_species, vertex_species].set(500.0)
rep_alpha_table = jnp.full((n_species, n_species), 2.5)
morse_narrow_alpha = 5.0

morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha).at[jnp.array([2, 3]), jnp.array([3, 2])].set(morse_narrow_alpha).at[jnp.array([4, 5]), jnp.array([5, 4])].set(morse_narrow_alpha)

def generate_idx_pairs(n_species):
    idx_pairs = []
    for i in range(1, n_species):
        for j in range(i + 1, n_species):
            idx_pairs.append((i, j))
    return idx_pairs

# Generate pairs of indices for the off-diagonal elements
generated_idx_pairs = generate_idx_pairs(n_species)

def make_tables(opt_params, num_monomers, use_custom_pairs=False, custom_pairs=None):
    morse_eps_table = jnp.full((n_species, n_species), small_value)
    
    if use_custom_pairs and custom_pairs is not None:
        idx_pairs = custom_pairs
    else:
        idx_pairs = generated_idx_pairs
    
    # Set off-diagonal elements
    for i, (idx1, idx2) in enumerate(idx_pairs):
        morse_eps_table = morse_eps_table.at[idx1, idx2].set(opt_params[i])
        morse_eps_table = morse_eps_table.at[idx2, idx1].set(opt_params[i])
    
    # Set diagonal elements excluding (0,0)
    if not use_custom_pairs:
        diagonal_start_idx = len(idx_pairs)
        for i in range(1, n_species):
            morse_eps_table = morse_eps_table.at[i, i].set(opt_params[diagonal_start_idx + i - 1])
    
    return morse_eps_table

def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params, num_monomers)
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
    morse_r0 = 0.0
    morse_rcut = 8.0 / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, 
                              alpha=morse_alpha, r0=morse_r0, ron=morse_rcut / 2.0)

morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)), in_axes=(0, None, 0, None, None))

def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)

inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))

def get_nmer_energy_fn(n):
    pairs = jnp.array(np.array(list(itertools.combinations(np.arange(n), 2))))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        pos_slices = [(i*9, (i+1)*9) for i in range(n)]
        species_slices = [(i*3, (i+1)*3) for i in range(n)]

        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        all_species = jnp.stack([jnp.repeat(species[start:end], 3) for start, end in species_slices])

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params).sum()
            rep_energy = rep_func(all_pos[i], all_pos[j], all_species[i], all_species[j]).sum()
            return morse_energy + rep_energy

        all_pairwise_energies = vmap(pairwise_energy)(pairs)
        return all_pairwise_energies.sum()

    return nmer_energy_fn

dimer_energy = jit(get_nmer_energy_fn(2))
trimer_energy = jit(get_nmer_energy_fn(3))

def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(jnp.sqrt(2. * jnp.pi / (jnp.abs(evals[6:]) + 1e-12)))
    return zvib

def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, seed=0, nrandom=100000):
    key = random.PRNGKey(seed)
    Nbb = len(pos)
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)

    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((Nbb * 6,), 0.)
        return nu0.at[3:6].set(angles)

    def ftilde(nu):
        q_tilde = jnp.matmul(evecs.T[6:].T, nu[6:])
        nu_tilde = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return utils.add_variables_all(q_tilde, nu_tilde)

    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))
    nu_fn = lambda nu: jnp.abs(jnp.linalg.det(jacfwd(ftilde)(nu)))
    Js = vmap(nu_fn)(nus)
    J = jnp.mean(Js)
    Jtilde = 8.0 * (jnp.pi ** 2) * J
    return Jtilde

def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib

mon_sigma = 3
dimer_sigma = data['dimer_sigma']
trimer_sigma = data['trimer_sigma']

mon_energy_fn = lambda q, pos, species, opt_params: 0.0
zrot_mod_sigma_mon = compute_zrot_mod_sigma(mon_energy_fn, mon_rb, mon_shape, data['mon_pc_species'][1], patchy_vals)
zvib_mon = 1.0
boltzmann_weight = 1.0
z_mon = compute_zc(boltzmann_weight, zrot_mod_sigma_mon, zvib_mon, mon_sigma, V)
z_mons = jnp.full(len(data['mon_pc_species']), z_mon)
log_z_mons = jnp.log(z_mons)

zrot_mod_sigma_dim = compute_zrot_mod_sigma(dimer_energy, dimer_rb, dimer_shape, jnp.array([1, 0, 2, 3, 0, 4]), patchy_vals)
zrot_mod_sigma_trim = compute_zrot_mod_sigma(trimer_energy, trimer_rb, trimer_shape, jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6]), patchy_vals)

dimer_pc_species = data['dimer_pc_species']
trimer_pc_species = data['trimer_pc_species']

def get_log_z_all(opt_params):
    def compute_log_z_dimer(dimer_species, dimer_sigma):
        zvib_dim = compute_zvib(dimer_energy, dimer_rb, dimer_shape, dimer_species, opt_params)
        e0 = dimer_energy(dimer_rb, dimer_shape, dimer_species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_dim = compute_zc(boltzmann_weight, zrot_mod_sigma_dim, zvib_dim, dimer_sigma, V)
        return jnp.log(z_dim)

    log_z_dims = vmap(compute_log_z_dimer)(dimer_pc_species, dimer_sigma)

    def compute_log_z_trimer(trimer_species, trimer_sigma):
        zvib_trim = compute_zvib(trimer_energy, trimer_rb, trimer_shape, trimer_species, opt_params)
        e0 = trimer_energy(trimer_rb, trimer_shape, trimer_species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_trim = compute_zc(boltzmann_weight, zrot_mod_sigma_trim, zvib_trim, trimer_sigma, V)
        return jnp.log(z_trim)

    log_z_trims = vmap(compute_log_z_trimer)(trimer_pc_species, trimer_sigma)

    log_z_all = jnp.concatenate([log_z_mons, log_z_dims, log_z_trims])
    return log_z_all

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

# Dynamically load and concatenate monomer counts
monomer_counts = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    mon_key = f"{letter}_mon_counts"
    dim_key = f"{letter}_dimer_counts"
    tri_key = f"{letter}_trimer_counts"
    
    if mon_key in data and dim_key in data and tri_key in data:
        monomer_counts.append(jnp.concatenate([data[mon_key], data[dim_key], data[tri_key]]))

nper_structure = jnp.array(monomer_counts)

def loss_fn(log_concs_struc, log_z_list, opt_params):
    conc_params_start_idx = len(patchy_vals)
    m_conc = opt_params[conc_params_start_idx:]
    tot_conc = jnp.sum(m_conc)
    log_mon_conc = safe_log(m_conc)
    
    def mon_loss_fn(mon_idx):
        mon_val = jnp.log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
        return mon_val - log_mon_conc[mon_idx]

    def struc_loss_fn(struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        def get_vcs_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_vca = jnp.log(V) + log_concs_struc[mon_idx]
            return n_sa * log_vca

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(num_monomers)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(num_monomers)).sum()

        return log_vcs - vcs_denom - log_zs + z_denom
    
    mon_loss = vmap(mon_loss_fn)(jnp.arange(num_monomers))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(num_monomers, tot_num_structures))
    combined_loss = jnp.concatenate([mon_loss, struc_loss])
    loss_var = jnp.var(combined_loss)
    loss_max = jnp.var(combined_loss)

    tot_loss = jnp.linalg.norm(combined_loss) + loss_var + 10 * loss_max
    return tot_loss, combined_loss, loss_var

def optimality_fn(log_concs_struc, log_z_list, opt_params):
    return grad(lambda log_concs_struc, log_z_list, opt_params: loss_fn(log_concs_struc, log_z_list, opt_params)[0])(log_concs_struc, log_z_list, opt_params)

inner_solver_logs = []

@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    gd = GradientDescent(fun=lambda log_concs_struc, log_z_list, opt_params: loss_fn(log_concs_struc, log_z_list, opt_params)[0], maxiter=6000, implicit_diff=True)
    sol = gd.run(init_guess, log_z_list, opt_params)
    
    final_params = sol.params
    final_loss, combined_losses, loss_var = loss_fn(final_params, log_z_list, opt_params)
    max_loss = jnp.max(combined_losses)
    second_max_loss = jnp.partition(combined_losses, -2)[-2]
    
    inner_solver_logs.append({
        "final_loss": final_loss,
        "loss_var": loss_var,
        "max_loss": max_loss,
        "second_max_loss": second_max_loss,
        "combined_losses": combined_losses 
    })
    
    return final_params

def ofer(opt_params, target_indices, desired_yields, num_monomers):
    log_z_list = get_log_z_all(opt_params)
    conc_params_start_idx = len(patchy_vals)
    tot_conc = jnp.sum(opt_params[conc_params_start_idx:])
    struc_concs_guess = jnp.full(tot_num_structures, safe_log(tot_conc / tot_num_structures))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    
    def compute_target_yield(target_idx):
        return safe_log(yields[target_idx])
    
    target_yields = vmap(compute_target_yield)(target_indices)
    return target_yields

def ofer_grad_fn(opt_params):
    target_yields = ofer(opt_params, jnp.array(inx_targets), jnp.array(desired_yields), num_monomers)
    losses = jnp.abs(jnp.array(desired_yields) - jnp.exp(target_yields))
    return jnp.sum(losses), losses

def project(param, num_monomers):
    conc_min, conc_max = 0.0001, 0.2
    concs = jnp.clip(param[-num_monomers:], a_min=conc_min, a_max=conc_max)
    total_conc = jnp.sum(concs)
    if total_conc > 0.5:
        concs = 0.5 * (concs / total_conc)
    param = param.at[-num_monomers:].set(concs)
    return param

our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
outer_optimizer = optax.adam(1e-2)
params = init_params
opt_state = outer_optimizer.init(params)

n_outer_iters = 150
outer_losses = []

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += [f"conc_{chr(ord('A') + i)}" for i in range(num_monomers)]

with open("morse_log.txt", "w") as log_file:
    log_file.write("Iteration\t" + "\t".join([f"Target_Yield_{target_shapes[i]}" for i in range(len(targets))]) + "\t" + "\t".join(param_names) + "\n")

    for i in tqdm(range(n_outer_iters)):
        (loss_val, losses), grads = our_grad_fn(params)
        outer_losses.append(loss_val)
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = project(params, num_monomers)
        
        log_file.write(f"{i+1}\t" + "\t".join(map(str, [desired_yields[i] - losses[i] for i in range(len(targets))])) + "\t" + "\t".join(map(str, params.tolist())) + "\n")
        
        print(f"Iteration {i+1}/{n_outer_iters}")
        print(f"Loss: {loss_val}")
        for idx, target in enumerate(targets):
            print(f"Yield{idx+1}: {desired_yields[idx] - losses[idx]}")
        print(f"Param: {params}")
        print(f"Concentrations: {params[-num_monomers:]}")
        print(f"Gradients: {grads}")

final_params = params
final_target_yields = ofer(final_params, jnp.array(inx_targets), jnp.array(desired_yields), num_monomers)

final_params_dict = {name: final_params[idx] for idx, name in enumerate(param_names)}

print(f"Final Optimized Parameters:")
for name, value in final_params_dict.items():
    print(f"{name}: {value}")

print(f"Final Target Yields: {final_target_yields}")

pdb.set_trace()
