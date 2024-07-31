import numpy as np
import pickle
import time
import jax.numpy as jnp
import optax
from jax import random, vmap, hessian, jacfwd, jit, value_and_grad, grad, lax, checkpoint
from tqdm import tqdm
from jax_md import space
import potentials
import utils
from jax_transformations3d import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
from checkpoint import checkpoint_scan
import pdb
import functools
import itertools
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
SEED = 42
key = random.PRNGKey(SEED)


targets = [
    {"structure": [1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10], "desired_yield": 0.5},
]

target_shapes = [t["structure"] for t in targets]

use_custom_pairs = True
custom_pairs = [(2, 3), (4, 5), (6, 7), (8, 9)]

def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('arvind_test.pkl')

species1 = data['1_pc_species']
species2 = data['2_pc_species']
species3 = data['3_pc_species']
species4 = data['4_pc_species']
species5 = data['5_pc_species']

n_1 = species1.shape[0]
n_2 = species2.shape[0]
n_3 = species3.shape[0]
n_4 = species4.shape[0]
n_5 = species5.shape[0]

tot_num_structures = n_1 + n_2 + n_3 + n_4 + n_5

num_monomers = 5

def indx_of_target(target):
    target = jnp.array(target)
    target_reversed = target[::-1]
    
    if target.shape[0] == 1 * 3:
        for i in range(n_1):
            if jnp.array_equal(species1[i], target) or jnp.array_equal(species1[i], target_reversed):
                return i
                
    elif target.shape[0] == 2 * 3:
        for i in range(n_2):
            if jnp.array_equal(species2[i], target) or jnp.array_equal(species2[i], target_reversed):
                return i + n_1
                
    elif target.shape[0] == 3 * 3:
        for i in range(n_3):
            if jnp.array_equal(species3[i], target) or jnp.array_equal(species3[i], target_reversed):
                return i + n_1 + n_2
            
    elif target.shape[0] == 3 * 4:
        for i in range(n_4):
            if jnp.array_equal(species4[i], target) or jnp.array_equal(species4[i], target_reversed):
                return i + n_1 + n_2 + n_3
            
    elif target.shape[0] == 3 * 5:
        for i in range(n_5):
            if jnp.array_equal(species5[i], target) or jnp.array_equal(species5[i], target_reversed):
                return i + n_1 + n_2 + n_3 +n_5

inx_targets = [indx_of_target(t["structure"]) for t in targets]
desired_yields = [t["desired_yield"] for t in targets]

euler_scheme = "sxyz"


V = 1250.0
kT = 1.0
n = 5  # number of monomers

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
patchy_vals = jnp.full(num_monomers-1, 6.0)  # FIXME for optimization over specific attraction strengths
#patchy_vals = jnp.full(n_morse_vals, 4.0) 
# Generate initial concentrations dynamically based on the number of monomers
initial_concentrations = jnp.full(num_monomers, 0.15)
concs = initial_concentrations

init_params = jnp.concatenate([patchy_vals, concs])

def make_shape(size):
    base_shape = jnp.array([
        [-a, 0., b],  # first patch
        [-a, b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],  # second patch
        [-a, -b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],
        [0., 0., a],
        [0., a * jnp.cos(jnp.pi / 6.), -a * jnp.sin(jnp.pi / 6.)],  # second sphere
        [0., -a * jnp.cos(jnp.pi / 6.), -a * jnp.sin(jnp.pi / 6.)],  # third sphere
        [a, 0., b],  # first patch
        [a, b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)],  # second patch
        [a, -b * jnp.cos(jnp.pi / 6.), -b * jnp.sin(jnp.pi / 6.)]  # third patch
    ], dtype=jnp.float64)
    return jnp.array([base_shape for _ in range(size)])

def make_rb(size, separation=2.0, noise=1e-14):
    if size == 1:
        return jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)

    rand_vals = random.normal(key, shape=(size,))

    rb = []
    half_size = size // 2

    for i in range(size):
        if size % 2 == 0:
            if i < half_size:
                rb.extend([-separation / 2.0 * (size - 1 - 2 * i), rand_vals[i] * noise, 0, 0, 0, 0])
            else:
                rb.extend([separation / 2.0 * (size - 1 - 2 * (size - 1 - i)), rand_vals[i] *  noise, 0, 0, 0, 0])
        else:
            if i == half_size:
                rb.extend([0, 0, 0, 0, 0, 0])
            elif i < half_size:
                rb.extend([-separation * (half_size - i), rand_vals[i] * noise, 0, 0, 0, 0])
            else:
                rb.extend([separation * (i - half_size), rand_vals[i] * noise, 0, 0, 0, 0])

    return jnp.array(rb, dtype=jnp.float64)

shape1 = make_shape(1)
shape2 = make_shape(2)
shape3 = make_shape(3)
shape4 = make_shape(4)
shape5 = make_shape(5)

rb1 = make_rb(1)
rb2 = make_rb(2)
rb3 = make_rb(3)
rb4 = make_rb(4)
rb5 = make_rb(5)

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

def make_tables(opt_params, num_monomers, use_custom_pairs=True, custom_pairs=custom_pairs):
    morse_eps_table = jnp.full((n_species, n_species), 3.0)
    morse_eps_table = morse_eps_table.at[0, :].set(small_value)
    morse_eps_table = morse_eps_table.at[:, 0].set(small_value)
    
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

energy2 = jit(get_nmer_energy_fn(2))
energy3 = jit(get_nmer_energy_fn(3))
energy4 = jit(get_nmer_energy_fn(4))
energy5 = jit(get_nmer_energy_fn(5))

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

sigma1 = 3
sigma2 = data['2_sigma']
sigma3 = data['3_sigma']
sigma4 = data['4_sigma']
sigma5 = data['5_sigma']


#pdb.set_trace()

mon_energy_fn = lambda q, pos, species, opt_params: 0.0
zrot_mod_sigma_1 = compute_zrot_mod_sigma(mon_energy_fn, rb1, shape1, data['1_pc_species'][1], patchy_vals,1)
zvib_1 = 1.0
boltzmann_weight = 1.0
z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigma1, V)
z_1s = jnp.full(len(data['1_pc_species']), z_1)
log_z_1 = jnp.log(z_1s)


zrot_mod_sigma_2 = compute_zrot_mod_sigma(energy2, rb2, shape2, jnp.array([1, 0, 2, 1, 0, 2]), patchy_vals)
zrot_mod_sigma_3 = compute_zrot_mod_sigma(energy3, rb3, shape3, jnp.array([1, 0, 2, 1, 0, 2, 1, 0, 2]), patchy_vals)
zrot_mod_sigma_4 = compute_zrot_mod_sigma(energy4, rb4, shape4, jnp.array([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2]), patchy_vals)
zrot_mod_sigma_5 = compute_zrot_mod_sigma(energy5, rb5, shape5, jnp.array([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2]), patchy_vals)

def compute_log_z_5(species, sigma):
    zvib_5 = compute_zvib(energy5, rb5, shape5, species, opt_params)
    e0 = energy4(rb5, shape5, species, opt_params)
    boltzmann_weight = jnp.exp(-e0 / kT)
    z_5 = compute_zc(boltzmann_weight, zrot_mod_sigma_5, zvib_5, sigma, V)
    return jnp.log(z_5)

def get_log_z_all(opt_params):
    
    def compute_log_z_2(species, sigma):
        zvib_2 = compute_zvib(energy2, rb2, shape2, species, opt_params)
        e0 = energy2(rb2, shape2, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_2 = compute_zc(boltzmann_weight, zrot_mod_sigma_2, zvib_2, sigma, V)
        return jnp.log(z_2)

    log_z_2 = vmap(compute_log_z_2)(species2, sigma2)

    def compute_log_z_3(species, sigma):
        zvib_3 = compute_zvib(energy3, rb3, shape3, species, opt_params)
        e0 = energy3(rb3, shape3, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_3 = compute_zc(boltzmann_weight, zrot_mod_sigma_3, zvib_3, sigma, V)
        return jnp.log(z_3)

    log_z_3 = vmap(compute_log_z_3)(species3, sigma3)
    
    def compute_log_z_4(species, sigma):
        zvib_4 = compute_zvib(energy4, rb4, shape4, species, opt_params)
        e0 = energy4(rb4, shape4, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_4 = compute_zc(boltzmann_weight, zrot_mod_sigma_4, zvib_4, sigma, V)
        return jnp.log(z_4)

    log_z_4 = vmap(compute_log_z_4)(species4, sigma4) 

    def compute_log_z_5(species, sigma):
        zvib_5 = compute_zvib(energy5, rb5, shape5, species, opt_params)
        e0 = energy4(rb5, shape5, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z_5 = compute_zc(boltzmann_weight, zrot_mod_sigma_5, zvib_5, sigma, V)
        return jnp.log(z_5)

    compute_log_z_5 = checkpoint(compute_log_z_5)

    assert species5.shape[0] == sigma5.shape[0], "species5 and sigma5 must have the same length."

    flat_species5 = species5.reshape(species5.shape[0], -1)
    xs = jnp.concatenate([flat_species5, sigma5[:, None]], axis=-1)

    def scan_fn(carry, x):
        flat_species, sigma = x[:-1], x[-1]
        species = flat_species.reshape(species5.shape[1:])
        result = compute_log_z_5(species, sigma)
        return carry, result

    checkpoint_freq = 10 

    scan_with_ckpt = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_freq)

    _, log_z_5 = scan_with_ckpt(scan_fn, None, xs)
    log_z_5 = jnp.array(log_z_5)

    log_z_all = jnp.concatenate([log_z_1, log_z_2, log_z_3, log_z_4, log_z_5])
    return log_z_all

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

# Dynamically load and concatenate monomer counts
monomer_counts = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    key1 = f"{letter}_1_counts"
    key2 = f"{letter}_2_counts"
    key3 = f"{letter}_3_counts"
    key4 = f"{letter}_4_counts"
    key5 = f"{letter}_5_counts"
    
    if key1 in data and key2 in data and key3 in data and key4 in data and key5 in data:
        monomer_counts.append(jnp.concatenate([data[key1], data[key2], data[key3], data[key4], data[key5]]))

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
    conc_min, conc_max = 0.00001, 0.2
    concs = jnp.clip(param[-num_monomers:], a_min=conc_min, a_max=conc_max)
    total_conc = jnp.sum(concs)
    if total_conc > 0.5:
        concs = 0.5 * (concs / total_conc)
    param = param.at[-num_monomers:].set(concs)
    return param


params = init_params
num_params = len(params)
mask = jnp.zeros(num_params)
mask = mask.at[-num_monomers:].set(1.0)

def masked_grads(grads):
    return grads * mask

our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 100
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
        # Apply the mask to the gradients
        grads = masked_grads(grads)
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = project(params, num_monomers)
        
        log_file.write(f"{i+1}\t" + "\t".join(map(str, [np.sqrt(desired_yields[i]**2 - losses[i]**2) for i in range(len(targets))])) + "\t" + "\t".join(map(str, params.tolist())) + "\n")
        
        print(f"Iteration {i+1}/{n_outer_iters}")
        print(f"Loss: {loss_val}")
        for idx, target in enumerate(targets):
            print(f"Yield{idx+1}: {np.sqrt(desired_yields[idx]**2 - losses[idx]**2)}")
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

