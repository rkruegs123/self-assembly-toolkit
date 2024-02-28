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

mon_pc_species = data['mon_pc_species']
dimer_pc_species = data['dimer_pc_species']
trimer_pc_species = data['trimer_pc_species']

#species_list = jnp.concatenate([mon_pc_species, dimer_pc_species, trimer_pc_species]) 

# mon count for each configuration 
A_mon_counts = data['A_mon_counts']
A_dimer_counts = data['A_dimer_counts']
A_trimer_counts = data['A_trimer_counts']

B_mon_counts = data['B_mon_counts']
B_dimer_counts = data['B_dimer_counts']
B_trimer_counts = data['B_trimer_counts']


A_count = jnp.concatenate([A_mon_counts, A_trimer_counts, A_dimer_counts])#fixme
B_count = jnp.concatenate([B_mon_counts, B_trimer_counts, B_dimer_counts])#fixme
copies_per_structure = jnp.array([A_count, B_count])

V = 1.
n = copies_per_structure.shape[0]
s = A_count.shape[0]


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


def pairwise_repulsion(ipos, jpos, i_species, j_species):
  
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)

    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)
               
                     

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

def trimer_energy(q, pos, species):
    
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
    

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2))
    tot_energy += jnp.sum(morse_func(pos1, pos3, species1, species3))
    tot_energy += jnp.sum(morse_func(pos2, pos3, species2, species3))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2)) 
    tot_energy += jnp.sum(rep_func(pos1, pos3, species1, species3))  
    tot_energy += jnp.sum(rep_func(pos2, pos3, species2, species3))  

    return tot_energy  


def dimer_energy(q, pos, species):
    
    positions = get_positions(q, pos)
    
    pos1 = positions[:9] 
    pos2 = positions[9:] 
                 
    species1 = species[:3]  
    species2 = species[3:]
    species1 = onp.repeat(species1, 3) 
    species2 = onp.repeat(species2, 3)
    

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2))               

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


def calculate_zc_mon(kBT=1, V=1, seed=0, nrandom=30000):

    key = random.PRNGKey(seed)


    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((1 * 6,), 0.) #Maybe Fixme
        return nu0.at[3:6].set(angles)

 
    def f(nu):
        return nu

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
    


def calculate_zc(energy_fn, q, pos, species, kBT=1, V=1, seed=0, nrandom=100000):

    zvib = get_zvib(energy_fn, q, pos, species)

    
    E0 = energy_fn(q, pos, species)
    boltzmann_weight = jnp.exp(-E0/kBT)
    n_mon = pos.shape[0]
    #sigma = 3**(n_mon-1)
    #print( sigma)
    sigma = 1.

    return boltzmann_weight * V * (1/sigma) * zvib


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


def run( z_rot, z_mon, seed=0):

    Zc_mon = jnp.repeat(z_mon, n)
    Zc_dimer = process_part(dimer_energy, dimer_rb, dimer_shapes, dimer_pc_species, chunk_size=10)
    Zc_trimer = process_part(trimer_energy, trimer_rb, trimer_shapes, trimer_pc_species, chunk_size=10)

                                                                              
    
    Zc_dimer = Zc_dimer *  z_rot[0]
    Zc_dimer = Zc_trimer *  z_rot[1]
    Zc_all= jnp.concatenate([Zc_mon, Zc_trimer, Zc_dimer]) #fixme
    log_zc_list= jnp.log(Zc_all)
    return log_zc_list


#V = 1.
#n = copies_per_structure.shape[0]
#s = A_count.shape[0]



def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))


def ofer(log_zc_list, conc):
    
    log_mon_conc = safe_log(conc[0:n])
    log_mon_zc = log_zc_list[0:n]
    
    def loss_fn(log_structure_concentrations):

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
                log_zalpha = log_zc_list[mon_idx]
                return n_sa * log_zalpha

            z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

            diff = log_vcs - vcs_denom - log_zs + z_denom
            return jnp.abs(diff)
        
        monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
        structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s)) 
        total_loss =  structure_loss.sum() +  monomer_loss.sum()
        return total_loss 
    
    init_struct_concentrations = jnp.full(s, safe_log(conc.sum() / s))
    
    # Define optimizer
    optimizer = optax.adam(1e-2)  # Consider tuning the learning rate
    grad_fn = jit(value_and_grad(loss_fn))

    # Optimization loop
    losses = []
    params = init_struct_concentrations
    opt_state = optimizer.init(params)
    for i in range(1000):  # You might want to adjust the number of iterations
        loss, grads = grad_fn(params)
        losses.append(loss)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
    return params, losses

                              
def optimize_loss(z_rot, z_mon, initial_mon_concentrations, lr=1e-3, num_iters=100):
    
    def calculate_target_yield(z_rot, z_mon, mon_concentrations):
        target = 0.6
        Z = run(z_rot, z_mon)
        log_structure_concentrations = ofer(Z, mon_concentrations)[0]
        print("conce:", log_structure_concentrations )
        exp_structure_concentrations = jnp.exp(log_structure_concentrations)
        print("exp:", exp_structure_concentrations )
        yield_of_target = jnp.abs(exp_structure_concentrations[-1] / exp_structure_concentrations.sum())
        return jnp.abs(target - yield_of_target)

    optimizer = optax.adam(lr)
    params = initial_mon_concentrations
    opt_state = optimizer.init(params)

    # The grad_fn should be defined outside the loop.
    grad_fn = jit(value_and_grad(calculate_target_yield))

    gradients = []

    # Now, the loop doesn't reinitialize the optimizer and params every iteration.
    optimizer = optax.adam(lr)
    params = initial_mon_concentrations
    opt_state = optimizer.init(params)
    for i in tqdm(range(num_iters)):

        pdb.set_trace()
        value, grads = grad_fn(z_rot, z_mon, params)
        gradients.append(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Debug prints
        print(f"Iteration {i+1}:")
        print(f"Gradients: {grads}")
        print(f"Parameters before update: {params}")
        params = optax.apply_updates(params, updates)
        print(f"Parameters after update: {params}")
        print("one loop completed")

    final_yield_diff = calculate_target_yield(z_rot, z_mon, params)

    return params, final_yield_diff, gradientss

    """
    for i in tqdm(range(num_iters)):

        value, grads = grad_fn(params)
        gradients.append(grads)  #
        

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
   

    for i in tqdm(range(num_iters)):
        try:
            value, grads = grad_fn(z_rot, z_mon, params)
            gradients.append(grads)  # 
            updates, opt_state = optimizer.update(grads, opt_state,z_rot, z_mon, params)
            params = optax.apply_updates(params, updates)
        except Exception as e:
            print("An error occurred during optimization:", e)
            break

    final_yield_diff = calculate_target_yield(z_rot, z_mon, params)

    return params, final_yield_diff, gradients
  
      """



def ofer_v2(log_zc_list, conc):
    
    log_mon_conc = safe_log(conc[0:n])
    log_mon_zc = log_zc_list[0:n]
    
    def loss_fn(log_structure_concentrations):

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
                log_zalpha = log_zc_list[mon_idx]
                return n_sa * log_zalpha

            z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

            diff = log_vcs - vcs_denom - log_zs + z_denom
            return jnp.abs(diff)
        
        monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
        structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s)) 
        total_loss =  structure_loss.sum() +  monomer_loss.sum()
        return total_loss 
    
    init_struct_concentrations = jnp.full(s, safe_log(conc.sum() / s))
    
    # Define optimizer
    optimizer = optax.adam(5e-2)  # Consider tuning the learning rate
    params = init_struct_concentrations
    opt_state = optimizer.init(params)
    grad_fn = jit(value_and_grad(loss_fn))

    n_iters = 10000

    """
    for _ in range(n_iters):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    return params[-1]
    """
        


    @jit
    def scan_fn(opt_info, idx):
        struct_concs, opt_state = opt_info
        loss, grads = grad_fn(struct_concs)
        updates, opt_state = optimizer.update(grads, opt_state)
        struct_concs = optax.apply_updates(struct_concs, updates)

        return (struct_concs, opt_state), loss

    fin_opt_info, losses = lax.scan(scan_fn, (params, opt_state), jnp.arange(n_iters))
    fin_concs, fin_opt_state = fin_opt_info

    yields = fin_concs / fin_concs.sum()

    return fin_concs[-1], losses


    
    


    

if __name__ == "__main__":


    # Testing taking derivatives through the concentration optimization
    Z = jnp.load("some_zs.npy")
    initial_mon_concs = jnp.array([1e-3, 2e-3])

    my_fn = lambda some_concs: ofer_v2(Z, some_concs)
    our_grad_fn = value_and_grad(my_fn, has_aux=True)
    our_grad_fn = jit(our_grad_fn)
    

    optimizer = optax.adam(1e-3)  # Consider tuning the learning rate
    params = initial_mon_concs
    opt_state = optimizer.init(params)

    n_outer_iters = 10
    for _ in tqdm(range(n_outer_iters)):
        (val, losses), grads = our_grad_fn(params)
        print(f"Yield: {val}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    
    pdb.set_trace()



    
    
    z_rot_dimer = get_zrot(dimer_energy, dimer_rb, dimer_shapes, dimer_pc_species[1])
    z_rot_trimer = get_zrot(trimer_energy, trimer_rb, trimer_shapes, trimer_pc_species[1])
    z_mon = calculate_zc_mon(kBT=1)                 
    z_rot = jnp.array([z_rot_dimer,z_rot_trimer ])
                       
    conc_A = 0.001
    conc_B = 0.002
    m_conc = jnp.array([conc_A, conc_B])

    optimized_params, opt_yield, gradients = optimize_loss(z_rot, z_mon, m_conc)

    print("Optimized Parameters:", optimized_params)
    print("Optimized Yield:", opt_yield_diff)


