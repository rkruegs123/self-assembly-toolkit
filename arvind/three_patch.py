import pdb
import numpy as onp
import jax.numpy as jnp
from copy import deepcopy
import itertools

from jax_md import rigid_body, energy, util, space, dataclasses
from jax_md.rigid_body import RigidPointUnion, union_to_points
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev, random

from utils import euler_scheme, convert_to_matrix
from transformations import transformations as jts
from itertools import combinations, permutations, chain
from jax.config import config
config.update("jax_enable_x64", True)

#note that concentation and number of monomers is specified for now

conc = .001
Ntot = 12
V = Ntot/conc



displacement_fn, shift_fn = space.free()

r_vertex = 1.0
r_patch = 0.1

vertex_species = 0
"""
monomer_types = [[i, i, i, 0, i + 1, i + 1, i + 1] for i in range(1, n_species-1)]
mirrored_types = [[i + 1, i + 1, i + 1, 0, i, i, i] for i in range(1, n_species-1)]
monomer_types.extend(mirrored_types)


dimer_combinations = itertools.product(monomer_types, repeat=2)
trimer_combinations = itertools.product(monomer_types, repeat=3)

target_combinations = [[1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 0, 4, 4, 4, 5, 5, 5, 0, 6, 6, 6],
                  [6, 6, 6, 0,5, 5, 5, 4, 4, 4, 0, 3, 3, 3, 2, 2, 2, 0, 1, 1, 1,]]

ot_combinations_dimer = [jnp.array(comb).flatten() for comb in dimer_combinations]
ot_combinations_trimer = [jnp.array(comb).flatten() for comb in trimer_combinations 
                  if list(jnp.array(comb).flatten()) not in target_species]
"""
# distance of the COM of three patches on one side of the sphere 
dist = r_patch * onp.sqrt(3)

# distance between center and the primary vertex (COM of the top patch) of the 3 patcheson y-z plane
a = dist/onp.sqrt(3)

# distance between center and the base of the position of the 3 patcheson y-z plane
b = a/2

def enhanced_powerset(iterable):
    # enhanced_powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (2,1) (1,3) (3,1) (2,3) (3,2) (1,2,3) (1,3,2) (2,1,3) (2,3,1) (3,1,2)(3,2,1)
    s = list(iterable)
    combos = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return chain.from_iterable(permutations(combo) for combo in combos)



rb_pos = jnp.array([
    #position of the left patches
    [-(onp.sqrt(r_vertex**2 - a**2) + r_patch), a, 0], 
    [-(onp.sqrt(r_vertex**2 - b**2) + r_patch), -b, dist/2],  
    [-(onp.sqrt(r_vertex**2 - b**2) + r_patch), -b, -dist/2],
    #position of the center sphere
    [0.0, 0.0, 0.0],
    #position of the right patches
    [(onp.sqrt(r_vertex**2 - a**2) + r_patch), a, 0], 
    [(onp.sqrt(r_vertex**2 - b**2) + r_patch), -b, dist/2],  
    [(onp.sqrt(r_vertex**2 - b**2) + r_patch), -b, -dist/2]
])


n_species = 7


small_value = 1e-12  # Small value to replace zeros to avoid nans


ss_eps_table = onp.full((n_species, n_species), small_value)  
ss_eps_table[vertex_species, vertex_species] = 500.0  
ss_eps_table = jnp.array(ss_eps_table)

ss_sigma_table = onp.full((n_species, n_species), small_value)  
ss_sigma_table[vertex_species, vertex_species] = 2*r_vertex
ss_sigma_table = jnp.array(ss_sigma_table)

pair_ss_energy_fn = energy.soft_sphere_pair(
    displacement_fn,
    sigma=ss_sigma_table,
    epsilon=ss_eps_table,
    species=n_species
)

"""
default_weak_eps = 1.0
default_strong_eps = 100.0
morse_eps_table = onp.full((n_species, n_species), default_weak_eps)

 
morse_eps_table[vertex_species, :] = 0.0
morse_eps_table[:, vertex_species] = 0.0

for i in [4, 5, 6]:
    for j in [7, 8, 9]:
        morse_eps_table[i, j] = default_strong_eps
        morse_eps_table[j, i] = default_strong_eps


for i in [10, 11, 12]:
    for j in [13, 14, 15]:
        morse_eps_table[i, j] = default_strong_eps
        morse_eps_table[j, i] = default_strong_eps

morse_eps_table = jnp.array(morse_eps_table)

"""
default_weak_eps = 0.0
morse_eps_table = onp.full((n_species, n_species), default_weak_eps)
morse_eps_table[vertex_species, :] = 0.0
morse_eps_table[:, vertex_species] = 0.0
default_strong_eps = 10.0
morse_eps_table[onp.array([2, 3, 4, 5]), onp.array([3, 2, 5, 4])] = default_strong_eps
morse_eps_table = jnp.array(morse_eps_table)
"""


default_weak_eps = 1.0
morse_eps_table = onp.full((n_species, n_species), default_weak_eps)
morse_eps_table[vertex_species, :] = 0.0
morse_eps_table[:, vertex_species] = 0.0
default_strong_eps = 100.0
morse_eps_table[onp.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]), onp.array([7, 8, 9, 4, 5, 6, 13, 14, 15, 10, 11, 12])] = default_strong_eps
morse_eps_table = jnp.array(morse_eps_table)

"""

pair_morse_energy_fn = energy.morse_pair(
    displacement_fn,
    sigma=1.,
    epsilon=morse_eps_table,
    alpha= 5,
    species=n_species
)


# Initialize ground state position
mon_sep =2 * (r_vertex + r_patch)

dimer_pos0 = jnp.array([
    [0.0, 0.0, 0.0],
    [mon_sep, 0.0, 0.0],
])
dimer_euler0 = jnp.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])

trimer_pos0 = jnp.array([
    [0.0, 0.0, 0.0],
    [mon_sep, 0.0, 0.0],
    [2 * mon_sep, 0.0, 0.0]
])
trimer_euler0 = jnp.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0]
])



dimer_gs_flattened = list()
for pos, euler in zip(dimer_pos0, dimer_euler0):
    dimer_gs_flattened += list(pos)
    dimer_gs_flattened += list(euler)
dimer_gs_flattened = jnp.array(dimer_gs_flattened)

trimer_gs_flattened = list()
for pos, euler in zip(trimer_pos0, trimer_euler0):
    trimer_gs_flattened += list(pos)
    trimer_gs_flattened += list(euler)
trimer_gs_flattened = jnp.array(trimer_gs_flattened)
"""
dimer_species = onp.array([1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12])
trimer_species = onp.array([1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15, 0, 16, 17, 18])

"""
dimer_species = onp.array([1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 0, 4, 4, 4])
trimer_species = onp.array([1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 0, 4, 4, 4, 5, 5, 5, 0, 6, 6, 6])

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


def hess(energy_fn, pos_flat, species):
    H = hessian(energy_fn)(pos_flat, species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

def get_zvib(energy_fn, pos_flat, species):
    evals, evecs = hess(energy_fn, pos_flat, species)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
    return zvib


def energy_fn(rb_flattened, point_species, **kwargs):
    rb = rb_flattened.reshape(-1, 6)
    transformation_matrices = vmap(convert_to_matrix)(rb)
    point_positions = vmap(jts.matrix_apply, (0, None))(transformation_matrices, rb_pos)
    point_positions = point_positions.reshape(-1, 3)

    return pair_morse_energy_fn(point_positions, species=point_species, **kwargs) \
        + pair_ss_energy_fn(point_positions, species=point_species, **kwargs)


def hess(energy_fn, pos_flat, species):
    H = hessian(energy_fn)(pos_flat, species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

def get_zvib(energy_fn, pos_flat, species):
    evals, evecs = hess(energy_fn, pos_flat, species)
    zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))
    return zvib
    

    
def get_zrot(energy_fn, pos_flat, species, seed=0, nrandom=100000):

    key = random.PRNGKey(seed)
    Nbb = pos_flat.shape[0] // 6

    evals, evecs = hess(energy_fn, pos_flat, species)

    # Generate random states
    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((Nbb * 6,), 0.0)
        return nu0.at[3:6].set(angles)

    # transformation of reference frame
    def ftilde(nu):
        q_tilde = jnp.matmul(evecs.T[6:].T, nu[6:])
        nu_tilde = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape) 
        return add_variables_all(q_tilde, nu_tilde)
    
    f = jit(ftilde)
    

    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))

    #pdb.set_trace()
    nu_fn = jit(lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu))))

    Js = vmap(nu_fn)(nus)
    J = jnp.mean(Js)
    
    Jtilde = 8.0 * (jnp.pi**2) * J
    
    return Jtilde

def get_ztot(energy_fn, pos_flat, species, V, sigma=1, kBT=1, seed=0, nrandom=100000):

    Jtilde = get_zrot(energy_fn, pos_flat, species)
    zvib = get_zvib(energy_fn, pos_flat, species)
    E0 = energy_fn(pos_flat, species)
    boltzmann_weight = jnp.exp(-E0/kBT)
    
    return boltzmann_weight * V * (Jtilde/sigma) * zvib

def calculate_yield(energy_fn, kBT=1):
    
    target_energies = [energy_fn(trimer_gs_flattened, species_comb) for species_comb in target_combinations]
    
    ot_energies_trimer = [energy_fn(trimer_gs_flattened, species_comb) for comb in ot_combinations_trimer]
    
    ot_energies_dimer = [energy_fn(dimer_gs_flattened, species_comb) for comb in ot_combinations_dimer]

   
    boltzmann_factors = jnp.array([jnp.exp(-E/kBT) for E in target_energies + ot_energies_trimer + ot_energies_dimer])

    Z = jnp.sum(boltzmann_factors)

    probabilities = boltzmann_factors / Z

 
    target_yield = jnp.sum(probabilities[:len(target_combinations)])

    return target_yield

    
    
print('trimer rotation', get_zrot(energy_fn, trimer_gs_flattened, trimer_species))
print('dimer rotation', get_zrot(energy_fn, dimer_gs_flattened, dimer_species))
print('trimer vibration', get_zvib(energy_fn, trimer_gs_flattened, trimer_species))
print('dimer vibration',get_zvib(energy_fn, dimer_gs_flattened, dimer_species))
print('trimer energy', energy_fn(trimer_gs_flattened, trimer_species))
print('dimer energy',energy_fn( dimer_gs_flattened, dimer_species))
pdb.set_trace()
    


