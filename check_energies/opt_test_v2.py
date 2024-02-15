import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as onp

import jax.numpy as jnp
from jax_md import rigid_body, energy, util, space, dataclasses
import optax
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev, random, lax
import potentials
from jax_transformations3d import jax_transformations3d as jts
import itertools
from copy import deepcopy
from functools import reduce
import pickle
from scipy.optimize import fsolve
import optax
from jaxopt import BFGS, objective, GradientDescent, ScipyMinimize, OptaxSolver


#from jax.config import config
#config.update("jax_enable_x64", True)


def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


data = load_species_combinations('species_test.pkl')

# species combinations for different configurations
mon_pc_species = data['mon_pc_species']
dimer_pc_species = data['dimer_pc_species']
trimer_pc_species = data['trimer_pc_species']

# mon count for each configuration 
A_mon_counts = data['A_mon_counts']
A_dimer_counts = data['A_dimer_counts']
A_trimer_counts = data['A_trimer_counts']

A_count = jnp.concatenate([A_mon_counts, A_dimer_counts, A_trimer_counts])




vertex_species = 0
n_species = 7

# Helper functions

euler_scheme = "sxyz"

def convert_to_matrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T,R)



# Define the dimer
num_building_blocks = 3


a = 1 # distance of the center of the spheres from the BB COM
b = .3 # distance of the center of the patches from the BB COM
shape1 = onp.array([[-a, 0., b], # first patch
    [-a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], # second patch
    [-a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], 
    [0., 0., a],
    [0., a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # second sphere
    [0., -a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # third sphere
    [a, 0., b], # first patch
    [a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], # second patch
    [a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)] # third patch
])

# these are the positions of the spheres within the building block
shape2 = shape1.copy()
shape3 = shape1.copy()

mon_shape = jnp.array([shape1])
trimer_shapes = jnp.array([shape1, shape2, shape3])
dimer_shapes = jnp.array([shape1, shape2])

separation = 2.
noise = 1e-14

mon_rb = jnp.array([0, 0, 0], dtype=jnp.float64)

trimer_rb = jnp.array([-separation, noise, 0, 0, 0, 0,   
                     0, 0, 0, 0, 0, 0,                
                     separation, noise, 0, 0, 0, 0],      
                    dtype=jnp.float64)

dimer_rb = jnp.array([-separation/2.0, noise, 0, 0, 0, 0,
                     separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)


vertex_radius = a
patch_radius = 0.2*a


def get_positions(q, ppos):
    Mat = []
    for i in range(len(ppos)):  
        qi = i * 6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(jts.matrix_apply(mat, ppos[i]))
    
    real_ppos = jnp.array(real_ppos)
    real_ppos = real_ppos.reshape(-1,3)

    return real_ppos 

def get_positions2(q, ppos):
    Mat = []
    for i in range(len(ppos)):  
        qi = i * 6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(jts.matrix_apply(mat, ppos[i]))


    return real_ppos 


"""
Defining values for potentials between all species

"""


# Setup soft-sphere repulsion between table values
small_value = 1e-12 

rep_A_table = onp.full((n_species, n_species), small_value)  
rep_A_table[vertex_species, vertex_species] = 500.0  
rep_A_table = jnp.array(rep_A_table)

rep_rmax_table = onp.full((n_species, n_species), 2*vertex_radius)  
rep_rmax_table = jnp.array(rep_rmax_table)

rep_stg_alpha = 2.5
rep_alpha_table = onp.full((n_species, n_species), rep_stg_alpha)
rep_alpha_table = jnp.array(rep_alpha_table)



# Setup morse potential between table values
default_weak_eps = small_value
morse_eps_table = onp.full((n_species, n_species), default_weak_eps)
default_strong_eps = 10.0
morse_eps_table[onp.array([1, 2, 3]), onp.array([1, 2, 3])] = default_strong_eps
morse_eps_table = jnp.array(morse_eps_table)

morse_weak_alpha = 1e-12 
morse_alpha_table = onp.full((n_species, n_species), morse_weak_alpha)
morse_strong_alpha = 5.0
morse_alpha_table[onp.array([2, 3, 4, 5]), onp.array([3, 2, 5, 4])] = morse_strong_alpha
morse_alpha_table = jnp.array(morse_alpha_table)

@jit
def pairwise_repulsion(ipos, jpos, i_species, j_species):
  
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)

    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)
               
                     
@jit
def pairwise_morse(ipos, jpos, i_species, j_species):
                     
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
  
    morse_r0 = 0.0                                   
    morse_rcut = 8. / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
                     
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, 
                   alpha=morse_alpha, r0=morse_r0, ron=morse_rcut/2.)                    
                                      
dimer_pos = get_positions(dimer_rb, dimer_shapes)
trimer_pos = get_positions(trimer_rb, trimer_shapes)    

@jit
def energy_tot(q, pos, species):
    ppos = get_positions(q, pos)
    species = onp.repeat(species, 3) 

    n_particles = len(ppos)
    n_per_monomer = 9 
    monomer = jnp.repeat(jnp.arange(n_particles // n_per_monomer), n_per_monomer)

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
    rep_func = vmap(vmap(pairwise_repulsion, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))

    morse_energy_matrix = morse_func(ppos, ppos, species, species)
    rep_energy_matrix = rep_func(ppos, ppos, species, species)

    inter_monomer_mask = monomer[:, None] != monomer[None, :]
    mask = inter_monomer_mask & ~jnp.eye(n_particles, dtype=bool)

    morse_energy_matrix = morse_energy_matrix * mask
    rep_energy_matrix = rep_energy_matrix * mask

    tot_energy = jnp.sum(jnp.triu(morse_energy_matrix)) + jnp.sum(jnp.triu(rep_energy_matrix))

    return tot_energy

  
def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb,Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans, angles))

def add_variables_all(mas, mbs):
    """
    Given two vectors of length (6*n,), 'add' them per building block according
    to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))

    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(
        mas_temp, mbs_temp), mas.shape)


def hess(energy_fn, q, pos, species):

    H = hessian(energy_fn)(q, pos, species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def get_zvib(energy_fn, q, pos, species):
    evals, evecs = hess(energy_fn, q, pos, species)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
    return zvib

def get_zrot(energy_fn, q, pos, species, seed=0, nrandom=100000):

    key = random.PRNGKey(seed)
    Nbb = len(pos) #to be changed

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

def calculate_zc_mon(energy_fn, q, pos, species, kBT=1, V=1, seed=0, nrandom=100000):

    key = random.PRNGKey(seed)
    Nbb = len(pos) #to be changed

    evals, evecs = hess(energy_fn, q, pos, species)


    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((2 * 6,), 0.) #Maybe Fixme
        return nu0.at[3:6].set(angles)

 
    def ftilde(nu):
        return nu
    
    f = ftilde

    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))


    nu_fn = lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu)))

    Js = vmap(nu_fn)(nus)


    J = jnp.mean(Js)
    Jtilde = 8.0 * (jnp.pi**2) * J
    
    zvib = 1.
    boltzmann_weight = 1.
    n_mon = pos.shape[0]
    sigma = 1

    return boltzmann_weight * V * (Jtilde/sigma) * zvib
    
    
    
    return Jtilde


def calculate_zc(energy_fn, q, pos, species, kBT=1, V=1, seed=0, nrandom=100000):

    zvib = get_zvib(energy_fn, q, pos, species)
    Jtilde = get_zrot(energy_fn, q, pos, species, seed, nrandom)
    
    E0 = energy_fn(q, pos, species)
    boltzmann_weight = jnp.exp(-E0/kBT)
    n_mon = pos.shape[0]
    #sigma = 3**(n_mon-1)
    #print( sigma)
    sigma = 1.

    return boltzmann_weight * V * (Jtilde/sigma) * zvib




def calculate_zc_part(energy_fn, q, pos, species_chunk):
    return vmap(calculate_zc, in_axes=(None, None, None, 0))(energy_fn, q, pos, species_chunk)

def process_part(energy_fn, q, pos, species, chunk_size=10):
    n = len(species)
    results = []
    for i in range(0, n, chunk_size):
        species_chunk = species[i:i+chunk_size]
        chunk_result = calculate_zc_part(energy_fn, q, pos, species_chunk)
        results.append(chunk_result)
    return jnp.concatenate(results, axis=0)

# Apply chunk processing
Zc_dimer = process_part(energy_tot, dimer_rb, dimer_shapes, dimer_pc_species, chunk_size=10)
Zc_trimer = process_part(energy_tot, trimer_rb, trimer_shapes, trimer_pc_species, chunk_size=10)
Zc_mon = vmap(calculate_zc_mon, in_axes=(None, None, None, 0))(energy_tot, mon_rb, mon_shape, mon_pc_species)

Zc_all= jnp.concatenate([Zc_mon, Zc_dimer, Zc_trimer])
#Zc_all_log= jnp.log(Zc_all)

A_count = jnp.concatenate([A_mon_counts, A_dimer_counts, A_trimer_counts])


#Zc_mon = jnp.array([1])
#Zc_dimer = vmap(dummy_zc, in_axes=(None, None, None, 0))(energy_tot, dimer_rb, dimer_shapes, dimer_pc_species)
#Zc_trimer = vmap(dummy_zc, in_axes=(None, None, None, 0))(energy_tot, trimer_rb, trimer_shapes, trimer_pc_species)


#Zc_all= jnp.concatenate([Zc_mon, Zc_dimer, Zc_trimer])
log_zc_list = jnp.log(Zc_all)


#mon_indecies = jnp.arange(3)
V = 1
conc = jnp.array([0.1])
copies_per_structure = jnp.array([A_count])

# Defined:
# number of monomoers: n
# number of structurs (including monomers): s
# conc: (n,)
# V: float
# log_zc_list: (s,)
# copies_per_structure: (n, s)

# first n strcutrs are the monomers
# last (n-s) structures are not the monomers
n = 1
log_mon_conc = jnp.log(conc[0:n])
log_mon_zc = log_zc_list[0:n]
s = Zc_all.shape[0]


    

def loss_fn(log_structure_concentrations):


    def monomer_loss_fn(monomer_idx):
        
        monomer_val =  jnp.log(jnp.dot(copies_per_structure, jnp.exp(log_structure_concentrations)))
        diff = monomer_val - log_mon_conc[monomer_idx]
        # rmse = jnp.sqrt((diff)**2)
        # return rmse
        return jnp.abs(diff)

    def structure_loss_fn(struct_idx):
        log_vcs = jnp.log(V) + log_structure_concentrations[struct_idx]

        def get_vcs_denom(mon_idx):
            n_sa = copies_per_structure[mon_idx][struct_idx]
            log_vca = jnp.log(V) + log_structure_concentrations[mon_idx]
            #log_vca = jnp.log(V) + log_mon_conc[mon_idx]
            return n_sa * log_vca
        
        vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()

        log_zs = log_zc_list[struct_idx]
        
        def get_z_denom(mon_idx):
            n_sa = copies_per_structure[mon_idx][struct_idx]
            log_zalpha = log_zc_list[mon_idx]
            return n_sa * log_zalpha
        z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

        diff = log_vcs - vcs_denom - log_zs + z_denom
        # rmse = jnp.sqrt(diff**2)
        # return rmse
        return jnp.abs(diff)

    monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
    structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s))
    total_loss = structure_loss.sum() + monomer_loss.sum()
    # return total_loss
    
    # total_loss = jnp.sum(monomer_loss) #+ jnp.sum(structure_loss)
    # total_loss = structure_loss.sum() + monomer_loss.sum()
    # return total_loss

    #return jnp.concatenate([monomer_loss, structure_loss]).sum()
    return total_loss

jit_loss_fn = jit(loss_fn)


def optimize_loss( initial_guess):
    solver = GradientDescent(fun=lambda x: jit_loss_fn(x), maxiter=50000)
    #solver = ScipyMinimize(fun=lambda x: loss_fn(x), maxiter=50000)

    result = solver.run(initial_guess)
    optimized_structure_concentrations = result.params
    return optimized_structure_concentrations

"""
uniform_conc = conc[0]/len(Zc_all)
#initial_guess = jnp.repeat(uniform_conc,8)
params_init = jnp.repeat(uniform_conc,8)

# Choose an optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Create the solver
solver = OptaxSolver(opt=optimizer, fun=loss_fn, maxiter=50000)

# Run optimization
pdb.set_trace()
result = solver.run(init_params=params_init)
pdb.set_trace()
print("Optimized Concentrations:", result)



def optimize_loss( initial_guess):
    scheduler = optax.exponential_decay(init_value=1e-2, transition_steps=100, decay_rate=0.95)
    optimizer = optax.adam(learning_rate=scheduler)
    solver = OptaxSolver(opt=optimizer, fun=jit_loss_fn, maxiter=50000)
    #solver = ScipyMinimize(fun=lambda x: loss_fn(x), maxiter=50000)

    result = solver.run(initial_guess)
    optimized_structure_concentrations = result.params
    return optimized_structure_concentrations
"""


# Assuming the last two configurations might be slightly more probable
uniform_conc = conc[0]/len(Zc_all)
initial_guess = jnp.repeat(uniform_conc,8)
# Initial guess for structure concentrations
uniform_conc = conc[0]/len(Zc_all)


pdb.set_trace()
optimized_concentrations = optimize_loss( initial_guess)
pdb.set_trace()
print("Optimized Concentrations:", optimized_concentrations)



                     
                     
