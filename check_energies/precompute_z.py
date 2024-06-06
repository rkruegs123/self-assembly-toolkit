import matplotlib.pyplot as plt
import numpy as onp
import pickle
import time
import jax.numpy as jnp
import optax
from jax import random, vmap, hessian, jacfwd
from tqdm import tqdm
from jax_md import space
import potentials
from jax_transformations3d import jax_transformations3d as jts

from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

# Load species
def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('AB_species_test2.pkl')

# Define constants
V = 1.0
kT = 1.0
n = 2  # number of monomers
tot_num_structures = data['mon_pc_species'].shape[0] + data['dimer_pc_species'].shape[0] + data['trimer_pc_species'].shape[0]
vertex_species = 0
n_species = 5
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

shape = onp.array([
    [-a, 0., b],  # first patch
    [-a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)],  # second patch
    [-a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], 
    [0., 0., a],
    [0., a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)],  # second sphere
    [0., -a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)],  # third sphere
    [a, 0., b],  # first patch
    [a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)],  # second patch
    [a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)]  # third patch
])
mon_shape = jnp.array([shape])
dimer_shape = jnp.array([shape, shape])
trimer_shape = jnp.array([shape, shape, shape])

mon_rb = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
dimer_rb = jnp.array([-separation/2.0, noise, 0, 0, 0, 0, separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)
trimer_rb = jnp.array([-separation, noise, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, separation, noise, 0, 0, 0, 0], dtype=jnp.float64)

# Energy helper functions
small_value = 1e-12
morse_eps_table = onp.full((n_species, n_species), small_value)
default_strong_eps = 10.0
morse_eps_table[onp.array([2, 3]), onp.array([3, 2])] = default_strong_eps
morse_eps_table = jnp.array(morse_eps_table)

morse_narrow_alpha = 5.0
morse_alpha_table = onp.full((n_species, n_species), morse_narrow_alpha)
morse_alpha_table[onp.array([2, 3]), onp.array([3, 2])] = morse_narrow_alpha
morse_alpha_table = jnp.array(morse_alpha_table)

def pairwise_morse(ipos, jpos, i_species, j_species):
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
    morse_r0 = 0.0                                   
    morse_rcut = 8.0 / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, alpha=morse_alpha, r0=morse_r0, ron=morse_rcut/2.0)

morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))

rep_A_table = onp.full((n_species, n_species), small_value)  
rep_A_table[vertex_species, vertex_species] = 500.0  
rep_A_table = jnp.array(rep_A_table)

rep_rmax_table = onp.full((n_species, n_species), 2*vertex_radius)  
rep_rmax_table = jnp.array(rep_rmax_table)

rep_strong_alpha = 2.5
rep_alpha_table = onp.full((n_species, n_species), rep_strong_alpha)
rep_alpha_table = jnp.array(rep_alpha_table)

def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)

inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))

# Energy functions for different sizes
def dimer_energy(q, pos, species):
    positions = get_positions(q, pos)  # 18 particles (9 per monomer)
    pos1 = positions[:9]  # First monomer
    pos2 = positions[9:]  # Last monomer
    species1 = species[:3]
    species2 = species[3:]
    species1 = onp.repeat(species1, 3)
    species2 = onp.repeat(species2, 3)
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2))               
    return tot_energy 

def trimer_energy(q, pos, species):
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
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2))
    tot_energy += jnp.sum(morse_func(pos1, pos3, species1, species3))
    tot_energy += jnp.sum(morse_func(pos2, pos3, species2, species3))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2)) 
    tot_energy += jnp.sum(rep_func(pos1, pos3, species1, species3))  
    tot_energy += jnp.sum(rep_func(pos2, pos3, species2, species3))  
    return tot_energy

# Functions for computing partition functions
def hess(energy_fn, q, pos, species):
    H = hessian(energy_fn)(q, pos, species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

def compute_zvib(energy_fn, q, pos, species):
    evals, evecs = hess(energy_fn, q, pos, species)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
    return zvib

def compute_zrot_mod_sigma(energy_fn, q, pos, species, seed=0, nrandom=100000):
    key = random.PRNGKey(seed)
    Nbb = len(pos)
    evals, evecs = hess(energy_fn, q, pos, species)

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
## Precompute the monomer partition functions. Note: only have to do once, then copy

mon_energy_fn = lambda q, pos, species: 0.0
zrot_mod_sigma_mon = compute_zrot_mod_sigma(
    mon_energy_fn, mon_rb, mon_shape,
    data['mon_pc_species'][1], seed=0, nrandom=100000)
zvib_mon = 1.0
boltzmann_weight = 1.0
z_mon = compute_zc(boltzmann_weight, zrot_mod_sigma_mon, zvib_mon, mon_sigma, V)
z_mons = jnp.full(len(data['mon_pc_species']), z_mon)
log_z_mons = jnp.log(z_mons)

# Precompute the dimer partition functions
zrot_mod_sigma_dim = compute_zrot_mod_sigma(
    dimer_energy, dimer_rb, dimer_shape,
    data['dimer_pc_species'][0], seed=0,
    nrandom=int(1e5))
z_dims = list()
for idx, dimer_species in enumerate(tqdm(data['dimer_pc_species'], desc="Computing dimer partition functions")):
    zvib_dim = compute_zvib(dimer_energy, dimer_rb, dimer_shape, dimer_species)
    e0 = dimer_energy(dimer_rb, dimer_shape, dimer_species)
    boltzmann_weight = jnp.exp(-e0/kT)
    z_dim = compute_zc(boltzmann_weight, zrot_mod_sigma_dim, zvib_dim, dimer_sigma[idx], V)
    z_dims.append(z_dim)
z_dims = jnp.array(z_dims)
log_z_dims = jnp.log(z_dims)

# Precompute the trimer partition functions
zrot_mod_sigma_trim = compute_zrot_mod_sigma(
    trimer_energy, trimer_rb, trimer_shape,
    data['trimer_pc_species'][0], seed=0,
    nrandom=int(1e5))
z_trims = list()
for idx, trimer_species in enumerate(tqdm(data['trimer_pc_species'], desc="Computing trimer partition functions")):
    zvib_trim = compute_zvib(trimer_energy, trimer_rb, trimer_shape, trimer_species)
    e0 = trimer_energy(trimer_rb, trimer_shape, trimer_species)
    boltzmann_weight = jnp.exp(-e0/kT)
    z_trim = compute_zc(boltzmann_weight, zrot_mod_sigma_trim, zvib_trim, trimer_sigma[idx], V)
    z_trims.append(z_trim)
z_trims = jnp.array(z_trims)
log_z_trims = jnp.log(z_trims)

# Combine all log_z values
log_z_all= jnp.concatenate([log_z_mons, log_z_trims, log_z_dims]) 
print(log_z_all)
# Save to file
with open('log_z_all.pkl', 'wb') as f:
    pickle.dump(log_z_all, f)