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


from jax.config import config
config.update("jax_enable_x64", True)


"""
Defining and finding all monomer species combinations

"""

monomers = {
    'A': [1, 0, 2],
    'B': [3, 0, 4],
    'C': [5, 0, 6],
}


monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}

all_monomers = {**monomers, **monomers_prime}

def flatten_and_compare(struct1, struct2):
    struct1_nums = sum([all_monomers[mon] for mon in struct1], [])
    struct2_nums = sum([all_monomers[mon] for mon in struct2], [])
    return struct1_nums == struct2_nums[::-1]

all_combinations = []
for r in range(1, 4):  
    all_combinations.extend(itertools.combinations_with_replacement(all_monomers.keys(), r))


filtered_combinations = []
for comb in all_combinations:
    if all(not flatten_and_compare(comb, existing_comb) for existing_comb in filtered_combinations):
        filtered_combinations.append(comb)
        
mon_list = []
dimer_list = []
trimer_list = []

for comb in filtered_combinations:
    if len(comb) == 1:  # Monomer
        mon_list.append(comb)
    elif len(comb) == 2:  # Dimer
        dimer_list.append(comb)
    elif len(comb) == 3:  # Trimer
        trimer_list.append(comb)
        
def count_monomers(combinations, monomer_name):
    """
    Count how many times a specific monomer and its mirrored version appear in each combination.
    """
    monomer_counts = []
    for comb in combinations:
        count = 0        
        for mon in comb:
            if mon == monomer_name or mon == f"{monomer_name}'":  
                count += 1
        
        monomer_counts.append(count)
    
    return monomer_counts

A_mon_counts = jnp.array(count_monomers(mon_list, 'A'))
A_dimer_counts = jnp.array(count_monomers(dimer_list, 'A'))
A_trimer_counts = jnp.array(count_monomers(trimer_list, 'A'))

B_mon_counts = jnp.array(count_monomers(mon_list, 'B'))
B_dimer_counts = jnp.array(count_monomers(dimer_list, 'B'))
B_trimer_counts = jnp.array(count_monomers(trimer_list, 'B'))

C_mon_counts = jnp.array(count_monomers(mon_list, 'C'))
C_dimer_counts = jnp.array(count_monomers(dimer_list, 'C'))
C_trimer_counts = jnp.array(count_monomers(trimer_list, 'C'))
        
       
def combination_to_string(comb):
    return ' '.join(comb)

mon_list = [combination_to_string(comb) for comb in mon_list]
dimer_list = [combination_to_string(comb) for comb in dimer_list]
trimer_list = [combination_to_string(comb) for comb in trimer_list]

def get_numeric_combination(comb_str):

    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])    
    return numeric_combination


mon_pc_species = jnp.array([get_numeric_combination(mon) for mon in mon_list])
dimer_pc_species = jnp.array([get_numeric_combination(dimer) for dimer in dimer_list])
trimer_pc_species = jnp.array([get_numeric_combination(trimer) for trimer in trimer_list])


"""
Defining shape of monomer, dimer, trimer structures

"""

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
#Zc_dimer = vmap(calculate_zc, in_axes=(None, None, None, 0))(energy_tot, dimer_rb, dimer_shapes, dimer_pc_species)
#Zc_trimer = vmap(calculate_zc, in_axes=(None, None, None, 0))(energy_tot, trimer_rb, trimer_shapes, trimer_pc_species)
#Zc_dimer
#pdb.set_trace()
Zc_all= jnp.concatenate([Zc_mon, Zc_dimer, Zc_trimer])
Zc_all_log= jnp.log(Zc_all)

A_count = jnp.concatenate([A_mon_counts, A_dimer_counts, A_trimer_counts])
B_count = jnp.concatenate([B_mon_counts, B_dimer_counts, B_trimer_counts])
C_count = jnp.concatenate([C_mon_counts, C_dimer_counts, C_trimer_counts])





def get_conc_eqs(conc, V, log_zc_list, copies_per_structure):
    # Note: we assume that the monomers of chain_idxs correspond to the the first n_chains
    # possible structures

    eqs = list()
    
    
    # First, construct one equation enforce conservation law
    def cons_law(bb_log_concs):
        return jnp.log(conc) - jnp.log(jnp.dot(jnp.array(copies_per_structure), jnp.exp(bb_log_concs)))

    eqs.append(deepcopy(cons_law))


    log_monomer_energy = log_zc_list[0:4] #the first 3 are the monomer strutures

    # Then, for each non-monomeric structure, add a mass-action constraint
    # Note: need a maker function to make functions unique
    def make_mass_action_eq(struct_idx, copies):
        def f(bb_log_concs):

            monomer_log_conc = bb_log_concs[0:4] #the first 3 are the monomer strctures
            log_monomer_zc = log_zc_list[0:4] #the first 3 are the monomer strctures

            s1 = copies * log_monomer_zc

            s2 = copies * (jnp.log(V) + monomer_log_conc)

            # FIXME: absolute value? How does fsolve handle negative values?
            return log_zc_list[struct_idx] - s1 - jnp.log(V) - bb_log_concs[struct_idx] + s2
        return f

    #for i, copies in enumerate(copies_per_structure): # need index to get conc
       # if copies == 1 and i == 0 or i == 1 or i == 2:
            #assert(i == 0 and copies == 1)
            #assert(i == 0 )
            #continue

        mass_action_eq = make_mass_action_eq(i, copies)
        eqs.append(deepcopy(mass_action_eq))

    return eqs # list of n_possible_structures equations, each of which takes bb_log_concs as an argument


def exponential_decay(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate ** (i / decay_steps)
  return schedule


def calc_concentrations(conc, V, zc_list, copies_per_structure, tol=1e-10, verbose=True):
    # Assuming `get_conc_eqs` is defined elsewhere and accessible
    eqs = get_conc_eqs(conc, V, zc_list, copies_per_structure)

    def loss(bb_log_concs):
        eq_vals = [jnp.abs(eq(bb_log_concs)) for eq in eqs]  # note the absolute values
        return jnp.sum(jnp.array(eq_vals))

    all_guesses = list()  # FIXME: populate list
    n_possible_structures = len(copies_per_structure)
    ## Guess with equal concentrations
    eq_conc_val = jnp.log(conc / n_possible_structures)
    eq_conc_guess = jnp.array([eq_conc_val for _ in range(n_possible_structures)])
    all_guesses.append(eq_conc_guess)

    all_opts = list()
    for init_params in all_guesses:
        num_epochs = int(1e6)
        step_size = 1e-2
        exp_decay = optax.exponential_decay(init_value=step_size, transition_steps=num_epochs, decay_rate=1e-12)

        optimizer = optax.adam(learning_rate=exp_decay, b1=0.9, b2=0.99, eps=1e-16)
        opt_state = optimizer.init(init_params)

        @jit
        def update(params, opt_state, step):
            value, grads = value_and_grad(loss)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, value

        params = init_params  # Initial parameters are now directly used

        # Loop over training epochs
        for epoch in range(num_epochs):
            params, opt_state, _loss = update(params, opt_state, epoch)
            if verbose and epoch % 10000 == 0:
                print(f"Epoch {epoch}, Loss {_loss}")

        final_params = params  # With optax, params are already updated
        eq_vals = [jnp.abs(eq(final_params)) for eq in eqs]
        all_opts.append((final_params, _loss))

    # Assuming final_params is desired to be in the original (not log) space
    return jnp.exp(final_params)


print(calc_concentrations(0.1, 1, Zc_all_log, A_count))

#print(energy_tot( trimer_rb, trimer_shapes, trimer_species))  
#print(get_zrot(energy_tot, trimer_rb, trimer_shapes, trimer_species)) 
#print(energy_tot( dimer_rb, dimer_shapes, dimer_species))  
#print(calculate_zc(energy_tot, dimer_rb, dimer_shapes, dimer_species)) 
#print(calculate_zc(energy_tot, trimer_rb, trimer_shapes,trier_species)) 

                     
                     