import argparse
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

# Argument parser setup
parser = argparse.ArgumentParser(description='Optimization script for self-assembly simulations.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--filename', type=str, default='arvind_test.pkl', help='Input data file name.')
parser.add_argument('--n_outer_iters', type=int, default=100, help='Number of outer iterations.')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for optimizer.')
parser.add_argument('--use_custom_pairs', type=bool, default=True, help='Flag to use custom pairs.')
parser.add_argument('--custom_pairs', type=str, default='[(2, 3), (4, 5), (6, 7), (8, 9)]', help='Custom pairs in the form of a list of tuples.')
parser.add_argument('--target_structure', type=str, default='[1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10]', help='Target structure as a list of integers.')
parser.add_argument('--desired_yield', type=float, default=0.5, help='Desired yield for the target structure.')
parser.add_argument('--small_value', type=float, default=1e-12, help='Small value for repulsive potential.')
parser.add_argument('--morse_narrow_alpha', type=float, default=5.0, help='Morse narrow alpha value.')
parser.add_argument('--large_morse_eps', type=float, default=6.0, help='Large Morse epsilon value for patchy particles.')
args = parser.parse_args()

SEED = args.seed
main_key = random.PRNGKey(SEED)

# Parse custom pairs and target structure from string inputs
custom_pairs = eval(args.custom_pairs)
target_structure = eval(args.target_structure)

# Example targets
targets = [
    {"structure": target_structure, "desired_yield": args.desired_yield},
]
use_custom_pairs = args.use_custom_pairs

def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the species data from a file
data = load_species_combinations(args.filename)

# Determine the number of monomers dynamically from the data
num_monomers = max(int(k.split('_')[0]) for k in data.keys() if k.endswith('_pc_species'))

# Initialize the species_data dictionary
species_data = {}
tot_num_structures = 0

# Populate the species_data dictionary
for i in range(1, num_monomers + 1):
    key = f"{i}_pc_species"
    species_data[key] = data[key]
    tot_num_structures += species_data[key].shape[0]

# Function to find the index of a target structure
def indx_of_target(target, species_data):
    target = jnp.array(target)
    target_reversed = target[::-1]
    num_monomers = len(species_data)
    
    offset = 0
    for i in range(1, num_monomers + 1):
        key = f"{i}_pc_species"
        current_species = species_data[key]
        for j in range(current_species.shape[0]):
            if jnp.array_equal(current_species[j], target) or jnp.array_equal(current_species[j], target_reversed):
                return j + offset
        offset += current_species.shape[0]
    
    return None  # Return None if the target is not found

# Get the indices of the target structures
inx_targets = [indx_of_target(t["structure"], species_data) for t in targets]
desired_yields = [t["desired_yield"] for t in targets]

# Example monomer counts
monomer_counts = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    counts_list = []
    for i in range(1, 6):  # Change the range if more counts are needed
        key = f"{letter}_{i}_counts"
        if key in data:
            counts_list.append(data[key])
    if counts_list:  # Only append if counts_list is not empty
        monomer_counts.append(jnp.concatenate(counts_list))

euler_scheme = "sxyz"

V = 1250.0
kT = 1.0
n = num_monomers  # number of monomers

# Shape and energy helper functions
a = 1.0  # distance of the center of the spheres from the BB COM
b = 0.3  # distance of the center of the patches from the BB COM
separation = 2.0
noise = 1e-14
vertex_radius = a
patch_radius = 0.2 * a
small_value = args.small_value
vertex_species = 0
n_patches = n * 2  # 2 species of patches per monomer type
n_species = n_patches + 1  # plus the common vertex species 0

n_morse_vals = n_patches * (n_patches - 1) // 2 + n_patches  # all possible pair permutations plus same patch attraction (i,i)
patchy_vals = jnp.full(n-1, args.large_morse_eps)  # FIXME for optimization over specific attraction strengths
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
    global main_key  # Use the global main_key
    if size == 1:
        return jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)

    main_key, subkey = random.split(main_key)  # Split the main_key properly
    rand_vals = random.normal(subkey, shape=(size,))

    rb = []
    half_size = size // 2

    for i in range(size):
        if size % 2 == 0:
            if i < half_size:
                rb.extend([-separation / 2.0 * (size - 1 - 2 * i), rand_vals[i] * noise, 0, 0, 0, 0])
            else:
                rb.extend([separation / 2.0 * (size - 1 - 2 * (size - 1 - i)), rand_vals[i] * noise, 0, 0, 0, 0])
        else:
            if i == half_size:
                rb.extend([0, 0, 0, 0, 0, 0])
            elif i < half_size:
                rb.extend([-separation * (half_size - i), rand_vals[i] * noise, 0, 0, 0, 0])
            else:
                rb.extend([separation * (i - half_size), rand_vals[i] * noise, 0, 0, 0, 0])

    return jnp.array(rb, dtype=jnp.float64)

sizes = range(1, num_monomers + 1)
shapes = {size: make_shape(size) for size in sizes}
rbs = {size: make_rb(size) for size in sizes}

rb1 = rbs[1]
shape1 = shapes[1]
sigma1 = data['1_sigma']

rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)
rep_A_table = jnp.full((n_species, n_species), small_value).at[vertex_species, vertex_species].set(500.0)
rep_alpha_table = jnp.full((n_species, n_species), 2.5)

morse_narrow_alpha = args.morse_narrow_alpha
morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)

def generate_idx_pairs(n_species):
    idx_pairs = []
    for i in range(1, n_species):
        for j in range(i + 1, n_species):
            idx_pairs.append((i, j))
    return idx_pairs

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

sizes = range(1, n+1)
shapes = {size: make_shape(size) for size in sizes}
rbs = {size: make_rb(size) for size in sizes}
sigmas = {size: data[f'{size}_sigma'] for size in sizes if f'{size}_sigma' in data}
energy_fns = {size: jit(get_nmer_energy_fn(size)) for size in range(2, 6)}

mon_energy_fn = lambda q, pos, species, opt_params: 0.0
zrot_mod_sigma_1 = compute_zrot_mod_sigma(mon_energy_fn, rbs[1], shapes[1], data['1_pc_species'][1], patchy_vals, 1)
zvib_1 = 1.0
boltzmann_weight = 1.0
z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1], V)
z_1s = jnp.full(len(data['1_pc_species']), z_1)
log_z_1 = jnp.log(z_1s)

zrot_mod_sigma_values = {}
for size in range(2, n + 1):
    zrot_mod_sigma_values[size] = compute_zrot_mod_sigma(energy_fns[size], rbs[size], shapes[size], jnp.array([1, 0, 2] * size), patchy_vals)

def get_log_z_all(opt_params, zrot_mod_sigma_values):
    def compute_log_z(size, species, sigma):
        energy_fn = energy_fns[size]
        shape = shapes[size]
        rb = rbs[size]
        zrot_mod_sigma = zrot_mod_sigma_values[size]
        zvib = compute_zvib(energy_fn, rb, shape, species, opt_params)
        e0 = energy_fn(rb, shape, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / kT)
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma, V)
        return jnp.log(z)
    
    log_z_all = []
       
    for size in range(2, n + 1):
        species = data[f'{size}_pc_species']
        sigma = data[f'{size}_sigma']
        
        compute_log_z_ckpt = checkpoint(lambda sp, sg: compute_log_z(size, sp, sg))
        flat_species = species.reshape(species.shape[0], -1)
        xs = jnp.concatenate([flat_species, sigma[:, None]], axis=-1)

        def scan_fn(carry, x):
            flat_species, sigma = x[:-1], x[-1]
            species_new = flat_species.reshape(species.shape[1:])
            result = compute_log_z_ckpt(species_new, sigma)
            return carry, result

        checkpoint_freq = 10
        scan_with_ckpt = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_freq)
        _, log_z = scan_with_ckpt(scan_fn, None, xs)
        log_z = jnp.array(log_z)

        log_z_all.append(log_z)
    
    log_z_all = jnp.concatenate(log_z_all)
    
    return log_z_all

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

# Example monomer counts
monomer_counts = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    counts_list = []
    for i in range(1, n+1):  
        key = f"{letter}_{i}_counts"
        if key in data:
            counts_list.append(data[key])
    if counts_list:  
        monomer_counts.append(jnp.concatenate(counts_list))

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
    log_z_list = get_log_z_all(opt_params, zrot_mod_sigma_values)
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
    conc_min, conc_max = 0.00001, 3
    concs = jnp.clip(param[-num_monomers:], a_min=conc_min, a_max=conc_max)
    return param

params = init_params
num_params = len(params)
mask = jnp.zeros(num_params)
mask = mask.at[-num_monomers:].set(1.0)

def masked_grads(grads):
    return grads * mask

our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
outer_optimizer = optax.adam(args.learning_rate)
opt_state = outer_optimizer.init(params)

n_outer_iters = args.n_outer_iters
outer_losses = []

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += [f"conc_{chr(ord('A') + i)}" for i in range(num_monomers)]

with open("morse_log.txt", "w") as log_file:
    log_file.write("Iteration\t" + "\t".join([f"Target_Yield_{targets[i]['structure']}" for i in range(len(targets))]) + "\t" + "\t".join(param_names) + "\n")

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