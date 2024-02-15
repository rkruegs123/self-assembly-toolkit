import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as onp
import jax.numpy as jnp
import optax
import potentials
from functools import reduce
from scipy.optimize import fsolve
import json
import optax
import pickle
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev, random, lax
from jaxopt import BFGS, objective, GradientDescent, ScipyMinimize, OptaxSolver
from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


####

# Defined:
# number of monomoers: n
# number of structurs (including monomers): s
# conc: (n,)
# V: float
# log_zc_list: (s,)
# copies_per_structure: (n, s)

# first n strcutrs are the monomers
# last (n-s) structures are not the monomers




def read_from_text(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(line.strip())  
    return data


def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('AB_species_test.pkl')

A_mon_counts = data['A_mon_counts']
A_dimer_counts = data['A_dimer_counts']
A_trimer_counts = data['A_trimer_counts']

B_mon_counts = data['B_mon_counts']
B_dimer_counts = data['B_dimer_counts']
B_trimer_counts = data['B_trimer_counts']

A_count = jnp.concatenate([A_mon_counts, A_dimer_counts, A_trimer_counts])
B_count = jnp.concatenate([B_mon_counts, B_dimer_counts, B_trimer_counts])

copies_per_structure = jnp.array([A_count, B_count])

#copies_per_structure = read_from_text('copies_per_structure.txt')
log_zc_list = read_from_text('log_zc_list.txt')
log_zc_list = [float(item) for item in log_zc_list]

log_zc_list = jnp.array(log_zc_list)


V = 1
conc = jnp.array([0.1, 0.1, 0.1])


def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def safe_exp(x, clip_value=88.0):  # np.log(np.finfo(np.float32).max)
    return jnp.exp(jnp.clip(x, a_min=None, a_max=clip_value))

n = 2
log_mon_conc = safe_log(conc[0:n])
log_mon_zc = log_zc_list[0:n]
s = log_zc_list.shape[0]    

def loss_fn(log_structure_concentrations):



    def monomer_loss_fn(monomer_idx):
        
        monomer_val = safe_log(jnp.dot(copies_per_structure, safe_exp(jnp.nan_to_num(log_structure_concentrations))))
        #monomer_val =  jnp.log(jnp.dot(copies_per_structure, jnp.exp(log_structure_concentrations)))
        diff = monomer_val - log_mon_conc[monomer_idx]
 
        # rmse = jnp.sqrt((diff)**2)
        # return rmse
        return jnp.abs(diff)

    def structure_loss_fn(struct_idx):
        log_vcs = safe_log(V) + log_structure_concentrations[struct_idx]

        def get_vcs_denom(mon_idx):
            n_sa = copies_per_structure[mon_idx][struct_idx]
            #n_sa = conc[mon_idx]
            #log_vca = safe_log(V) + log_structure_concentrations[mon_idx]
            log_vca = jnp.log(V) + log_mon_conc[mon_idx]
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

    monomer_loss = vmap(monomer_loss_fn)(jnp.arange(n))
    structure_loss = vmap(structure_loss_fn)(jnp.arange(n, s))
    total_loss = structure_loss.sum() + monomer_loss.sum()
    #total_loss = jnp.nan_to_num(total_loss)

    # return total_loss
    
    # total_loss = jnp.sum(monomer_loss) #+ jnp.sum(structure_loss)
    # total_loss = structure_loss.sum() + monomer_loss.sum()
    # return total_loss

    #return jnp.concatenate([monomer_loss, structure_loss]).sum()
    return total_loss

#jit_loss_fn = jit(loss_fn)



"""
def optimize_loss( initial_guess):
    solver = GradientDescent(fun=lambda x: loss_fn(x), maxiter=50000)
    #solver = ScipyMinimize(fun=lambda x: loss_fn(x), maxiter=50000)

    result = solver.run(initial_guess)
    optimized_structure_concentrations = result.params
    return optimized_structure_concentrations
"""
def optimize_loss( initial_guess):
    scheduler = optax.exponential_decay(init_value=1e-2, transition_steps=100, decay_rate=0.95)
    optimizer = optax.adam(learning_rate=scheduler)
    solver = OptaxSolver(opt=optimizer, fun=loss_fn, maxiter=50000)
    #solver = ScipyMinimize(fun=lambda x: loss_fn(x), maxiter=50000)

    result = solver.run(initial_guess)
    optimized_structure_concentrations = result.params
    return optimized_structure_concentrations


# Assuming the last two configurations might be slightly more probable
uniform_conc = conc[0]/s
initial_guess = jnp.repeat(uniform_conc,s)
# Initial guess for structure concentrations
uniform_conc = conc[0]/s


pdb.set_trace()
optimized_concentrations = optimize_loss(initial_guess)
pdb.set_trace()
print("Optimized Concentrations:", optimized_concentrations)

                     
                     


#print(energy_tot( trimer_rb, trimer_shapes, trimer_species))  
#print(get_zrot(energy_tot, trimer_rb, trimer_shapes, trimer_species)) 
#print(energy_tot( dimer_rb, dimer_shapes, dimer_species))  
#print(calculate_zc(energy_tot, dimer_rb, dimer_shapes, dimer_species)) 
#print(calculate_zc(energy_tot, trimer_rb, trimer_shapes,trier_species)) 

                    
                     