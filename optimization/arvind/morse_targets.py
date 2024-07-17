import matplotlib.pyplot as plt
import numpy as onp
import pickle
import time
import jax.numpy as jnp
import optax
from jax import random, vmap, hessian, jacfwd, jit, value_and_grad, grad
from tqdm import tqdm
from jax_md import space
import potentials
from jax_transformations3d import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
import pdb

from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

# Load species
def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('ABC_species.pkl')

# Define constants
V = 1.0
kT = 1.0
n = 3  # number of monomers
tot_num_structures = data['mon_pc_species'].shape[0] + data['dimer_pc_species'].shape[0] + data['trimer_pc_species'].shape[0]
print(tot_num_structures)


vertex_species = 0
n_species = 7
a = 1  # distance of the center of the spheres from the BB COM
b = .3  # distance of the center of the patches from the BB COM
vertex_radius = a
patch_radius = 0.2 * a

# Define helper functions
euler_scheme = "sxyz"

def convert_to_matrix(mi):
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T, R)

def get_positions(q, ppos):
    Mat = []
    for i in range(len(ppos)):
        qi = i * 6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(jts.matrix_apply(mat, ppos[i]))

    real_ppos = jnp.array(real_ppos)
    real_ppos = real_ppos.reshape(-1, 3)

    return real_ppos

def add_variables(ma, mb):
    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb, Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))
    return jnp.concatenate((trans, angles))

def add_variables_all(mas, mbs):
    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))
    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(mas_temp, mbs_temp), mas.shape)

# Define shapes
separation = 2.0
noise = 1e-14

shape = jnp.array([
    [-a, 0., b],  # first patch
    [-a, b*jnp.cos(jnp.pi/6.), -b*jnp.sin(jnp.pi/6.)],  # second patch
    [-a, -b*jnp.cos(jnp.pi/6.), -b*jnp.sin(jnp.pi/6.)], 
    [0., 0., a],
    [0., a*jnp.cos(jnp.pi/6.), -a*jnp.sin(jnp.pi/6.)],  # second sphere
    [0., -a*jnp.cos(jnp.pi/6.), -a*jnp.sin(jnp.pi/6.)],  # third sphere
    [a, 0., b],  # first patch
    [a, b*jnp.cos(jnp.pi/6.), -b*jnp.sin(jnp.pi/6.)],  # second patch
    [a, -b*jnp.cos(jnp.pi/6.), -b*jnp.sin(jnp.pi/6.)]  # third patch
])
mon_shape = jnp.array([shape])
dimer_shape = jnp.array([shape, shape])
trimer_shape = jnp.array([shape, shape, shape])

mon_rb = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
dimer_rb = jnp.array([-separation/2.0, noise, 0, 0, 0, 0, separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)
trimer_rb = jnp.array([-separation, noise, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, separation, noise, 0, 0, 0, 0], dtype=jnp.float64)

small_value = 1e-12
patchy_vals = jnp.full(21, 4.)

concA = 0.1
concB = 0.1
concC = 0.1

concs = jnp.array([concA, concB, concC])

init_params = jnp.concatenate([patchy_vals, concs])

#patchy_vals = jnp.array([1., 5.0, 500.0,  2.5]) # default_strong_eps, morse_narrow_alpha, A_rep, rep_strong_alpha
# Energy helper functions
rep_rmax_table = onp.full((n_species, n_species), 2*vertex_radius)  
rep_rmax_table = jnp.array(rep_rmax_table)

morse_narrow_alpha = 5.
morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)
morse_alpha_table = morse_alpha_table.at[jnp.array([2, 3]), jnp.array([3, 2])].set(morse_narrow_alpha)
morse_alpha_table = morse_alpha_table.at[jnp.array([4, 5]), jnp.array([5, 4])].set(morse_narrow_alpha)

rep_A_table = onp.full((n_species, n_species), small_value)  
rep_A_table[vertex_species, vertex_species] = 500.0  
rep_A_table = jnp.array(rep_A_table)

rep_strong_alpha = 2.5
rep_alpha_table = onp.full((n_species, n_species), rep_strong_alpha)
rep_alpha_table = jnp.array(rep_alpha_table)


def make_tables(opt_params):
    morse_eps_table = jnp.full((n_species, n_species), small_value)
    morse_eps_table = morse_eps_table.at[1, 2].set(opt_params[0])
    morse_eps_table = morse_eps_table.at[2, 1].set(opt_params[0])
    morse_eps_table = morse_eps_table.at[1, 3].set(opt_params[1])
    morse_eps_table = morse_eps_table.at[3, 1].set(opt_params[1])
    morse_eps_table = morse_eps_table.at[1, 4].set(opt_params[2])
    morse_eps_table = morse_eps_table.at[4, 1].set(opt_params[2])
    morse_eps_table = morse_eps_table.at[1, 5].set(opt_params[3])
    morse_eps_table = morse_eps_table.at[5, 1].set(opt_params[3])
    morse_eps_table = morse_eps_table.at[1, 6].set(opt_params[4])
    morse_eps_table = morse_eps_table.at[6, 1].set(opt_params[4])
    morse_eps_table = morse_eps_table.at[2, 3].set(opt_params[5])
    morse_eps_table = morse_eps_table.at[3, 2].set(opt_params[5])
    morse_eps_table = morse_eps_table.at[2, 4].set(opt_params[6])
    morse_eps_table = morse_eps_table.at[4, 2].set(opt_params[6])
    morse_eps_table = morse_eps_table.at[2, 5].set(opt_params[7])
    morse_eps_table = morse_eps_table.at[5, 2].set(opt_params[7])
    morse_eps_table = morse_eps_table.at[2, 6].set(opt_params[8])
    morse_eps_table = morse_eps_table.at[6, 2].set(opt_params[8])
    morse_eps_table = morse_eps_table.at[3, 4].set(opt_params[9])
    morse_eps_table = morse_eps_table.at[4, 3].set(opt_params[9])
    morse_eps_table = morse_eps_table.at[3, 5].set(opt_params[10])
    morse_eps_table = morse_eps_table.at[5, 3].set(opt_params[10])
    morse_eps_table = morse_eps_table.at[3, 6].set(opt_params[11])
    morse_eps_table = morse_eps_table.at[6, 3].set(opt_params[11])
    morse_eps_table = morse_eps_table.at[4, 5].set(opt_params[12])
    morse_eps_table = morse_eps_table.at[5, 4].set(opt_params[12])
    morse_eps_table = morse_eps_table.at[4, 6].set(opt_params[13])
    morse_eps_table = morse_eps_table.at[6, 4].set(opt_params[13])
    morse_eps_table = morse_eps_table.at[6, 5].set(opt_params[14])
    morse_eps_table = morse_eps_table.at[5, 6].set(opt_params[14])
    
    morse_eps_table = morse_eps_table.at[1, 1].set(opt_params[15])
    morse_eps_table = morse_eps_table.at[2, 2].set(opt_params[16])
    morse_eps_table = morse_eps_table.at[3, 3].set(opt_params[17])
    morse_eps_table = morse_eps_table.at[4, 4].set(opt_params[18])
    morse_eps_table = morse_eps_table.at[5, 5].set(opt_params[19])
    morse_eps_table = morse_eps_table.at[6, 6].set(opt_params[20])
    
    return morse_eps_table

def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params)
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
    morse_r0 = 0.0                                   
    morse_rcut = 8.0 / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
                     
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, 
                              alpha=morse_alpha, r0=morse_r0, ron=morse_rcut/2.0)

morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)), in_axes=(0, None, 0, None, None))

def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)

inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))

@jit
def dimer_energy(q, pos, species, opt_params):
    positions = get_positions(q, pos)  # 18 particles (9 per monomer)
    pos1 = positions[:9]  # First monomer
    pos2 = positions[9:]  # Last monomer
    species1 = species[:3]
    species2 = species[3:]
    species1 = onp.repeat(species1, 3)
    species2 = onp.repeat(species2, 3)
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2, opt_params))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2))               
    return tot_energy 

@jit
def trimer_energy(q, pos, species, opt_params):
    positions = get_positions(q, pos)  # 27 particles (9 per monomer)
    pos1 = positions[:9] 
    pos2 = positions[9:18]
    pos3 = positions[18:] 
    species1 = species[:3]  
    species2 = species[3:6]
    species3 = species[6:] 
    species1 = onp.repeat(species1, 3) 
    species2 = onp.repeat(species2, 3)
    species3 = onp.repeat(species3, 3)
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2, opt_params))
    tot_energy += jnp.sum(morse_func(pos1, pos3, species1, species3, opt_params))
    tot_energy += jnp.sum(morse_func(pos2, pos3, species2, species3, opt_params))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2)) 
    tot_energy += jnp.sum(rep_func(pos1, pos3, species1, species3))  
    tot_energy += jnp.sum(rep_func(pos2, pos3, species2, species3))  
    return tot_energy

def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
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
        return add_variables_all(q_tilde, nu_tilde)

    f = ftilde
    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))
    nu_fn = lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu)))
    Js = vmap(nu_fn)(nus)
    J = jnp.mean(Js)
    Jtilde = 8.0 * (jnp.pi**2) * J
    return Jtilde

def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib

mon_sigma = 3
dimer_sigma = data['dimer_sigma']
trimer_sigma = data['trimer_sigma']
# Precompute the monomer partition functions. Note: only have to do once, then copy

mon_energy_fn = lambda q, pos, species, opt_params: 0.0  # Added opt_params argument
zrot_mod_sigma_mon = compute_zrot_mod_sigma(
    mon_energy_fn, mon_rb, mon_shape,
    data['mon_pc_species'][1], patchy_vals)
zvib_mon = 1.0
boltzmann_weight = 1.0
z_mon = compute_zc(boltzmann_weight, zrot_mod_sigma_mon, zvib_mon, mon_sigma, V)
z_mons = jnp.full(len(data['mon_pc_species']), z_mon)
log_z_mons = jnp.log(z_mons)

# Precompute the rot partition functions
zrot_mod_sigma_dim = compute_zrot_mod_sigma(
    dimer_energy, dimer_rb, dimer_shape,
    jnp.array([1,0,2,3,0,4]),patchy_vals)

zrot_mod_sigma_trim = compute_zrot_mod_sigma(
    trimer_energy, trimer_rb, trimer_shape,
    jnp.array([1,0,2,3,0,4,5,0,6]), patchy_vals)

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

A_mon_counts = data['A_mon_counts']
A_dim_counts = data['A_dimer_counts']
A_trim_counts = data['A_trimer_counts']

B_mon_counts = data['B_mon_counts']
B_dim_counts = data['B_dimer_counts']
B_trim_counts = data['B_trimer_counts']

C_mon_counts = data['C_mon_counts']
C_dim_counts = data['C_dimer_counts']
C_trim_counts = data['C_trimer_counts']

A_count = jnp.concatenate([A_mon_counts, A_dim_counts, A_trim_counts])
B_count = jnp.concatenate([B_mon_counts, B_dim_counts,  B_trim_counts])
C_count = jnp.concatenate([C_mon_counts, C_dim_counts,  C_trim_counts])
nper_structure = jnp.array([A_count, B_count, C_count])



def loss_fn(log_concs_struc, log_z_list, opt_params):
    
    m_conc = jnp.array([opt_params[-3],opt_params[-2],opt_params[-1]])
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

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

        return log_vcs - vcs_denom - log_zs + z_denom
    
    mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(n, tot_num_structures))
    combined_loss = jnp.concatenate([mon_loss, struc_loss])
    loss_var = jnp.var(combined_loss)
    loss_max = jnp.var(combined_loss)

    tot_loss = jnp.linalg.norm(combined_loss) + loss_var + 10 * loss_max
    return tot_loss, combined_loss, loss_var

def optimality_fn(log_concs_struc, log_z_list, opt_params):
    return grad(lambda log_concs_struc, log_z_list, opt_params : loss_fn(log_concs_struc, log_z_list, opt_params)[0])(log_concs_struc, log_z_list, opt_params)

inner_solver_logs = []

@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    gd = GradientDescent(fun=lambda log_concs_struc, log_z_list, opt_params : loss_fn(log_concs_struc, log_z_list, opt_params)[0], maxiter=6000, implicit_diff=True)
    sol = gd.run(init_guess, log_z_list, opt_params=log_z_list)
    
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

y1 = 0.5
y2 = 0.2

def ofer(opt_params):
    
    log_z_list = get_log_z_all(opt_params)
    tot_conc  = jnp.sum(jnp.array([opt_params[-3],opt_params[-2],opt_params[-1]]))

    struc_concs_guess = jnp.full(tot_num_structures, safe_log(tot_conc / tot_num_structures))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)

    yields = fin_concs / jnp.sum(fin_concs)
    
    target_yield1 = safe_log(yields[1])
    target_yield2 = safe_log(yields[-1])
    yield_sum = safe_log(jnp.sum(yields))
    #monomerA = safe_log(yields[0])
    #monomerB = safe_log(yields[1])
    #monomerC = safe_log(yields[2])
    return target_yield1, target_yield2

def ofer_grad_fn(opt_params):
    target_yield1, target_yield2 = ofer(opt_params)
    loss1 = jnp.abs(y1 - jnp.exp(target_yield1))
    loss2 = jnp.abs(y2 - jnp.exp(target_yield2))
    #var_yield = jnp.var(jnp.array([target_yield1, target_yield2]))
    return loss1 + loss2, (loss1, loss2)

def project(param):
    return jnp.clip(param, a_min=1e-4)


our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))

outer_optimizer = optax.adam(1e-2)
params = init_params
opt_state = outer_optimizer.init(params)




n_outer_iters = 400
outer_losses = []

with open("morse_log.txt", "w") as log_file:
    log_file.write("Iteration\tTarget_Yield1\tTarget_Yield2\t" + "\t".join([f"Eps{j+1}" for j in range(24)]) + "\n")
    
    for i in tqdm(range(n_outer_iters)):
        (loss_val, (loss1, loss2)), grads = our_grad_fn(params)
        outer_losses.append(loss_val)
 
        
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = project(params)
        
        
        
        log_file.write(f"{i+1}\t{y1-loss1}\t{y2-loss2}\t" + "\t".join(map(str, params.tolist())) + "\n")
        
        print(f"Iteration {i+1}/{n_outer_iters}")
        print(f"Loss: {loss_val}")
        print(f"Yield1: {y1-loss1}")
        print(f"Yield1: {y2-loss2}")
        #print(f"Yield2: {jnp.exp(target_yield2)}")
        #print(f"Variance: {var_yield}")
        print(f"Param: {params}")
        print(f"Concentrations: {params[-3], params[-2], params[-1]}")
        print(f"Gradients: {grads}")

pdb.set_trace()
final_params = params
final_target_yield = ofer(final_params)[0]
print(f"Final Optimized Parameter (monomer concentration): {final_params}")
print(f"Final Target Yield: {final_target_yield}")
