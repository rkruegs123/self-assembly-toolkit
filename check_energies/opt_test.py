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
from jaxopt import BFGS, objective


from jax.config import config
config.update("jax_enable_x64", True)


def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the data
data = load_species_combinations('species_test.txt')

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


def dummy_zc(energy_fn, q, pos, species, kBT=1, V=1, seed=0, nrandom=100000):
    
    E0 = energy_fn(q, pos, species)
    boltzmann_weight = jnp.exp(-E0/kBT)
    n_mon = pos.shape[0]
    sigma = 1.

    return boltzmann_weight 


Zc_mon = jnp.array([1])
Zc_dimer = vmap(dummy_zc, in_axes=(None, None, None, 0))(energy_tot, dimer_rb, dimer_shapes, dimer_pc_species)
Zc_trimer = vmap(dummy_zc, in_axes=(None, None, None, 0))(energy_tot, trimer_rb, trimer_shapes, trimer_pc_species)


Zc_all= jnp.concatenate([Zc_mon, Zc_dimer, Zc_trimer])
log_zc_list = jnp.log(Zc_all)


#mon_indecies = jnp.arange(3)
V = 1
conc = jnp.array([0.1])
copies_per_structure = jnp.array(A_count)

# Defined:
# number of monomoers: n
# number of structurs (including monomers): s
# conc: (n,)
# V: float
# log_zc_list: (s,)
# copies_per_structure: (n, s)

n = 1

def loss_fn(structure_concentrations):
    # first n strcutrs are the monomers
    # last (n-s) structures are not the monomers
   
    s = len(structure_concentrations)
    log_mon_conc = jnp.log(conc[0:n])
    log_mon_zc = log_zc_list[0:n]
    
    #pdb.set_trace()

    def monomer_loss_fn(monomer_idx):
        
        monomer_val = jnp.log(monomer_idx) - jnp.dot(copies_per_structure[monomer_idx], jnp.exp(structure_concentrations))
        return abs(monomer_val - log_mon_conc[monomer_idx])

    def structure_loss_fn(struct_idx):
        
      #  def multiply(row_of_copies, single_log_monomer_zc):
          #  return row_of_copies * single_log_monomer_zc
        
        log_mon_zc_expanded = log_mon_zc[:, None]


        s1 = copies_per_structure * log_mon_zc_expanded
        
        #log_mon_zc_r = log_mon_zc[:, None]
        log_V_repeated = jnp.repeat(jnp.log(V), n)
        mon_term = log_V_repeated + log_mon_zc
        mon_term_expanded = mon_term[:, None]
        s2 = copies_per_structure * mon_term_expanded
        
       # s1 = vmap(multiply, in_axes=(0, 0))(copies_per_structure, log_mon_zc_r)
        #operation = lambda x, y: x * (jnp.log(V) + y)
        #s2 = vmap(operation, in_axes=(0, 0))(copies_per_structure, log_mon_conc[:, None])

        structure_val = log_zc_list[struct_idx] -  jnp.prod(s1) - jnp.log(V) - structure_concentrations[struct_idx] + jnp.prod(s2)
        return abs((V * structure_concentrations[struct_idx]) - structure_val)

    monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
    structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s))
    
    total_loss = jnp.sum(monomer_loss) + jnp.sum(structure_loss)

    return total_loss


def optimize_loss(n, initial_guess):
    solver = BFGS(fun=lambda x: loss_fn(x), maxiter=100)

    result = solver.run(initial_guess)
    optimized_structure_concentrations = result.params
    return optimized_structure_concentrations


# Assuming the last two configurations might be slightly more probable
initial_guess = jnp.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3])
  # Initial guess for structure concentrations
optimized_concentrations = optimize_loss(n, initial_guess)
print("Optimized Concentrations:", optimized_concentrations)


                     
                     