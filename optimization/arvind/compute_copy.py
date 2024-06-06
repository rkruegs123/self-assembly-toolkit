import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as onp
import csv
import argparse
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
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


data = load_species_combinations('AB_species_test.pkl')

max_dimer_species = 5
max_trimer_species = 2


mon_pc_species = data['mon_pc_species']
dimer_pc_species = data['dimer_pc_species'][-max_dimer_species:]
trimer_pc_species = data['trimer_pc_species'][:max_trimer_species]

#species_list = jnp.concatenate([mon_pc_species, dimer_pc_species, trimer_pc_species]) 

# mon count for each configuration 
A_mon_counts = data['A_mon_counts']
A_dimer_counts = data['A_dimer_counts'][-max_dimer_species:]
A_trimer_counts = data['A_trimer_counts'][:max_trimer_species]

B_mon_counts = data['B_mon_counts']
B_dimer_counts = data['B_dimer_counts'][-max_dimer_species:]
B_trimer_counts = data['B_trimer_counts'][:max_trimer_species]


A_count = jnp.concatenate([A_mon_counts, A_trimer_counts, A_dimer_counts])
B_count = jnp.concatenate([B_mon_counts, B_trimer_counts, B_dimer_counts])
copies_per_structure = jnp.array([A_count, B_count])



"""
Defining shape of monomer, dimer, trimer structures

"""

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

dimer_pos = get_positions(dimer_rb, dimer_shapes)
trimer_pos = get_positions(trimer_rb, trimer_shapes) 

"""
Defining values for potentials between all species

"""


# Setup soft-sphere repulsion between table values
def setup_rep_tables(rep_A, rep_stg_alpha):
    small_value = 1e-12
    rep_A_table = jnp.full((n_species, n_species), small_value)
    rep_A_table = rep_A_table.at[vertex_species, vertex_species].set(rep_A)
    rep_rmax_table = jnp.full((n_species, n_species), 2*vertex_radius)
    rep_alpha_table = jnp.full((n_species, n_species), rep_stg_alpha)

    return rep_A_table, rep_alpha_table, rep_rmax_table

def setup_morse_tables(morse_strong_eps, morse_strong_alpha):
    small_value = 1e-12
    default_weak_eps = small_value
    morse_eps_table = jnp.full((n_species, n_species), default_weak_eps)

    morse_eps_table = morse_eps_table.at[1, 1].set(morse_strong_eps)
    morse_eps_table = morse_eps_table.at[2, 2].set(morse_strong_eps)
    morse_eps_table = morse_eps_table.at[3, 3].set(morse_strong_eps)

    morse_weak_alpha = 1e-12
    morse_alpha_table = jnp.full((n_species, n_species), morse_weak_alpha)

    morse_alpha_table = morse_alpha_table.at[2, 3].set(morse_strong_alpha)
    morse_alpha_table = morse_alpha_table.at[3, 2].set(morse_strong_alpha)
    morse_alpha_table = morse_alpha_table.at[4, 5].set(morse_strong_alpha)
    morse_alpha_table = morse_alpha_table.at[5, 4].set(morse_strong_alpha)

    return morse_alpha_table, morse_eps_table



def pairwise_repulsion(tables, ipos, jpos, i_species, j_species):
    
    rep_A_table, rep_alpha_table, rep_rmax_table = tables[0], tables[1], tables[2]
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)

    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)
                                   

def pairwise_morse(tables, ipos, jpos, i_species, j_species):
    
    morse_alpha_table, morse_eps_table = tables[0], tables[1]
    
    morse_alpha = morse_alpha_table[i_species, j_species]   
    morse_d0 = morse_eps_table[i_species, j_species]
    
    morse_r0 = 0.0                                   
    morse_rcut = 8. / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
                     
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, 
                   alpha=morse_alpha, r0=morse_r0, ron=morse_rcut/2.)                    
                       

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


#pdb.set_trace()
#@jit
def trimer_energy(rep, morse, q, pos, species):
    
    positions = get_positions(q, pos)
    
    
    pos1 = positions[:9] 
    pos2 = positions[9:18]
    pos3 = positions[18:] 
                 
    species1 = species[:3]  
    species2 = species[3:6]
    species3 = species[6:12]
    species1 = onp.repeat(species1, 3) 
    species2 = onp.repeat(species2, 3)
    species3 = onp.repeat(species3, 3)
    
    rep_A, rep_stg_alpha = rep[0], rep[1]
    morse_eps, morse_alpha = morse[0], morse[1]
    
    tables_morse = setup_morse_tables(rep_A, rep_stg_alpha)
    tables_rep = setup_rep_tables( morse_eps, morse_alpha)

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, None, 0, None, 0)), in_axes=(None, 0, None, 0, None))
    tot_energy = jnp.sum(morse_func( tables_morse, pos1, pos2, species1, species2))
    tot_energy += jnp.sum(morse_func(tables_morse, pos1, pos3, species1, species3))
    tot_energy += jnp.sum(morse_func( tables_morse, pos2, pos3, species2, species3))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(None, 0, None, 0, None))
    tot_energy += jnp.sum(rep_func(tables_rep, pos1, pos2, species1, species2)) 
    tot_energy += jnp.sum(rep_func(tables_rep, pos1, pos3, species1, species3))  
    tot_energy += jnp.sum(rep_func(tables_rep, pos2, pos3, species2, species3))  

    return tot_energy  

#@jit
def dimer_energy(rep, morse, q, pos, species):
    
    positions = get_positions(q, pos)
    
    pos1 = positions[:9] 
    pos2 = positions[9:] 
                 
    species1 = species[:3]  
    species2 = species[3:]
    species1 = onp.repeat(species1, 3) 
    species2 = onp.repeat(species2, 3)
    
    rep_A, rep_stg_alpha = rep[0], rep[1]
    morse_eps, morse_alpha = morse[0], morse[1]
    
    tables_morse = setup_morse_tables(rep_A, rep_stg_alpha)
    tables_rep = setup_rep_tables( morse_eps, morse_alpha)

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, None, 0, None, 0)), in_axes=(None, 0, None, 0, None))
    tot_energy = jnp.sum(morse_func(tables_morse, pos1, pos2, species1, species2))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(None, 0, None, 0, None))
    tot_energy += jnp.sum(rep_func(tables_rep, pos1, pos2, species1, species2))               

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


def hess(energy_fn, rep, morse, q, pos, species):

    energy_hessian = hessian(energy_fn, argnums=2) 
    H = energy_hessian(rep, morse, q, pos, species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def get_zvib(energy_fn, rep, morse, q, pos, species):
    evals, evecs = hess(energy_fn, rep, morse, q, pos, species)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
    return zvib

def get_zrot(energy_fn, rep, morse, q, pos, species, seed=0, nrandom=100000):

    key = random.PRNGKey(seed)
    Nbb = len(pos) 

    evals, evecs = hess(energy_fn, rep, morse, q, pos, species)


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

@jit
def calculate_zc_mon( q, pos, species, kBT=1, V=1, seed=0, nrandom=100000):

    key = random.PRNGKey(seed)


    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((1 * 6,), 0.) #Maybe Fixme
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
    sigma = 1
    

    return  boltzmann_weight * V * (Jtilde/sigma) * zvib
    
    
    
    return Jtilde

@jit
def calculate_zc(energy_fn, rep, morse, q, pos, species, kBT=1, V=1, seed=0, nrandom=30000):

    zvib = get_zvib(energy_fn, rep, morse, q, pos, species)
    Jtilde = get_zrot(energy_fn, rep, morse, q, pos, species, seed, nrandom)
    
    E0 = energy_fn(rep, morse, q, pos, species)
    boltzmann_weight = jnp.exp(-E0/kBT)
    n_mon = pos.shape[0]
    #sigma = 3**(n_mon-1)
    #print( sigma)
    sigma = 1.

    return boltzmann_weight * V * (Jtilde/sigma) * zvib


def calculate_zc_part(energy_fn, rep, morse, q, pos, species_chunk):
    return vmap(calculate_zc, in_axes=(None, None, None, None, None, 0))(energy_fn, rep, morse, q, pos, species_chunk)

def process_part(energy_fn, rep, morse, q, pos, species, chunk_size=10):
    n = len(species)
    results = []
    for i in range(0, n, chunk_size):
        species_chunk = species[i:i+chunk_size]
        chunk_result = calculate_zc_part(energy_fn, rep, morse, q, pos, species_chunk)
        results.append(chunk_result)
    return jnp.concatenate(results, axis=0)

def process_part_fori_loop(energy_fn, rep, morse, q, pos, species, chunk_size=10):
    n = len(species)

    def body_fun(i, results):
        species_chunk = species[i:i+chunk_size]
        chunk_result = calculate_zc_part(energy_fn, rep, morse, q, pos, species_chunk)
        results = results.at[i // chunk_size].set(chunk_result)  # Assume results pre-allocated with correct shape
        return results

    num_batches = (n + chunk_size - 1) // chunk_size  # Calculate the number of batches
    results = jnp.zeros((num_batches,))  # Placeholder, adjust the shape according to actual output of calculate_zc_part
    results = lax.fori_loop(0, n, body_fun, results)

    return results.flatten() 

def run(args, seed=0):
    
    rep = jnp.array([args['rep_A'],args['rep_stg_alpha']])
    morse = jnp.array([args['morse_strong_eps'],args['morse_strong_alpha']])

   
    Zc_mon = vmap(calculate_zc_mon, in_axes=(None, None, 0))(mon_rb, mon_shape, mon_pc_species)
    Zc_dimer =  vmap(calculate_zc, in_axes=(None, None, None, None, None, 0))(dimer_energy, rep, morse, dimer_rb, dimer_shapes, dimer_pc_species)

    Zc_trimer = vmap(calculate_zc, in_axes=(None, None, None, None, None, 0))(trimer_energy, rep, morse, trimer_rb, trimer_shapes, trimer_pc_species)
    #Zc_dimer = process_part_fori_loop(dimer_energy, rep, morse, dimer_rb, dimer_shapes, dimer_pc_species, chunk_size=10)
    #Zc_trimer = process_part_fori_loop(trimer_energy, rep, morse, trimer_rb, trimer_shapes, trimer_pc_species, chunk_size=10)
    Zc_all= jnp.concatenate([Zc_mon, Zc_trimer, Zc_dimer])
    log_zc_list= jnp.log(Zc_all)
    return log_zc_list



#A_count = jnp.concatenate([A_mon_counts, A_dimer_counts, A_trimer_counts]) uncomment if there are 3 types of monomers 
#B_count = jnp.concatenate([B_mon_counts, B_dimer_counts, B_trimer_counts]) uncomment if there are 3 types of monomers 
copies_per_structure = jnp.array([A_count, B_count])

def get_argparse():
    parser = argparse.ArgumentParser(description='Compute the yield of a simple monomer chain system')
    parser.add_argument('--morse_strong_eps', type=float, default=10.0, help='Epsilon parameter for Morse potential')
    parser.add_argument('--morse_strong_alpha', type=float, default=5.0, help='Alpha parameter for Morse potential')
    parser.add_argument('--rep_A', type=float, default=500.0, help='A parameter for repulsion potential')
    parser.add_argument('--rep_stg_alpha', type=float, default=2.5, help='Alpha parameter for repulsion potential')
    parser.add_argument('--conc', type=float, default=0.001, help='Monomer concentration')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations for optimization')
    
    return parser

V = 1.
n = copies_per_structure.shape[0]
s = A_count.shape[0]
conc = jnp.repeat(0.001,n)

target_yield = 0.5

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def safe_exp(x, clip_value=88.0):  # np.log(np.finfo(np.float32).max)
    return jnp.exp(jnp.clip(x, a_min=None, a_max=clip_value))


def yield_fn(morse_strong_eps, morse_strong_alpha, log_structure_concentrations):


    log_zc_list = run(args)
    log_mon_conc = safe_log(conc[0:n])
    log_mon_zc = log_zc_list[0:n]

    def monomer_loss_fn(monomer_idx):

        monomer_val =  jnp.log(jnp.dot(copies_per_structure[monomer_idx],
                                       jnp.exp(log_structure_concentrations)))
        diff = monomer_val - log_mon_conc[monomer_idx]

        return jnp.abs(diff)

    def structure_loss_fn(struct_idx):
        log_vcs = jnp.log(V) + log_structure_concentrations[struct_idx]

        def get_vcs_denom(mon_idx):
            n_sa = copies_per_structure[mon_idx][struct_idx]
            log_vca = jnp.log(V) + log_structure_concentrations[mon_idx]

            return n_sa * log_vca

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()

        log_zs = log_zc_list[struct_idx]

        def get_z_denom(mon_idx):
            n_sa = copies_per_structure[mon_idx][struct_idx]
            #n_sa = conc[mon_idx]
            log_zalpha = log_zc_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

        diff = log_vcs - vcs_denom - log_zs + z_denom
        return jnp.abs(diff)

    exp_structure_concentrations = jnp.exp(log_structure_concentrations)
    yield_loss = jnp.abs(exp_structure_concentrations[-1]/exp_structure_concentrations.sum()-target_yield)
    monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
    structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s))
    total_loss =  structure_loss.sum() +  monomer_loss.sum() + yield_loss 

    return total_loss


def optimize_loss(args, initial_guess, lr, num_iters):

    morse_strong_eps = args['morse_strong_eps']
    morse_strong_alpha = args['morse_strong_alpha']

    optimizer = optax.adam(learning_rate=lr)
    params = {'concentrations': initial_guess}


    opt_state = optimizer.init(params)

    def total_loss_fn(params):
        return yield_fn(morse_strong_eps, morse_strong_alpha, params['concentrations'])

    grad_fn = jit(value_and_grad(total_loss_fn))

    pdb.set_trace()

    losses = []
    for i in tqdm(range(num_iters)):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        if i % 100 == 0 or i == num_iters - 1:
            print(f"Iteration {i}, Loss: {loss}")

    return params, losses




if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())
        
    
    

    uniform_conc = 2 * conc.sum() / s
    initial_guess = jnp.repeat(uniform_conc, s)

    optimized_params, losses = optimize_loss(args, initial_guess, args['lr'], args['num_iters'])
    print("Optimized Parameters:", optimized_params)



