import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as onp

import jax.numpy as jnp
from jax_md import rigid_body, energy, util, space, dataclasses
import optax
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev, random
import potentials
from jax_transformations3d import jax_transformations3d as jts

from jax.config import config
config.update("jax_enable_x64", True)


"""
Notes:
- we will consider only the dimer system
"""


vertex_species = 0
n_species = 6

# Helper functions

euler_scheme = "sxyz"

def convert_to_matrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T,R)

r_vertex = 1.0
r_patch = 0.1

# Define the dimer
num_building_blocks = 2

# distance of the COM of three patches on one side of the sphere 
dist = r_patch * onp.sqrt(3)

# distance between center and the primary vertex (COM of the top patch) of the 3 patcheson y-z plane
a = dist/onp.sqrt(3)

# distance between center and the base of the position of the 3 patcheson y-z plane
b = a/2

shape1 = onp.array([    
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

# these are the positions of the spheres within the building block
shape2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                jnp.array([1, 0, 0], dtype=jnp.float64)),
                          shape1
                         )
shape2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                jnp.array([0, 1, 0], dtype=jnp.float64)),
                          shape2
)
shapes = jnp.array([shape1, shape2])


separation = 2.
noise = 1e-15
rb_info = jnp.array([-separation/2.0, noise, 0, 0, 0, 0,
                     separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)

vertex_radius = r_vertex
patch_radius = r_patch

def get_positions(q, ppos):
    Mat = []
    for i in range(num_building_blocks):
        qi = i*6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i in range(num_building_blocks):
        real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

    return real_ppos

#get_positions(rb_info, shapes)


#points = get_positions(rb_info, shapes)
#points = onp.array(points).reshape(-1,3)
target_species = [ 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 0, 4, 4, 4]
target_species = jnp.array(target_species)
#pdb.set_trace()

test_rs_morse = onp.linspace(-0.25, 1.0, 100)
morse_d0 = 10.0
morse_a = 5.0
morse_r0 = 0.0
morse_rcut = 8. / morse_a + morse_r0
morsex_energies = potentials.morse_x(
    test_rs_morse, rmin=0, rmax=morse_rcut,
    D0=morse_d0,
    alpha=morse_a, r0=morse_r0,
    ron=morse_rcut/2.)


## Soft sphere


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
morse_alpha_table[onp.array([1, 2, 3]), onp.array([1, 2, 3])] = morse_strong_alpha
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
                     
                     
                     
def get_energy(q, pos, species):
    
    positions = get_positions(q, pos)

    pos1 = positions[0]  
    pos2 = positions[1]  

    species1 = species[:7]  
    species2 = species[7:]  

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
    Nbb = 2 #to be changed

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

      
print(get_energy(rb_info, shapes, target_species))
print(get_zrot(get_energy, rb_info, shapes, target_species))

"""
monomers = {
    'Red': [1, 0, 2],
    'Blue': [3, 0, 4],
    'Green': [5, 0, 6],
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
       
        
def combination_to_string(comb):
    return " ".join(comb)

mon_list = [combination_to_string(comb) for comb in mon_list]
dimer_list = [combination_to_string(comb) for comb in dimer_list]
trimer_list = [combination_to_string(comb) for comb in trimer_list]


def get_numeric_combination(comb_str):

    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])    
    return numeric_combination
"""
#specific dimer and trimer energy functions (can be used to check if energy from general function 
# is correct)

"""                     
def trimer_energy(q, pos, species):
    
    positions = get_positions2(q, pos)
    
    
    pos1 = positions[0] 
    pos2 = positions[1]
    pos3 = positions[2] 
                 
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
    
    pos1 = positions[0] 
    pos2 = positions[1] 
                 
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
    

monomers_tuples = [
    (1, 0, 2),
    (3, 0, 4),
    (5, 0, 6),
]

monomers_tuples += [tuple(reversed(monomer)) for monomer in monomers_tuples]

all_combinations_tuples = []
for r in range(1, 4):  
    all_combinations_tuples.extend(itertools.combinations_with_replacement(monomers_tuples, r))

def is_combination_mirrored(new_comb, existing_combs):
    new_comb_flat = sum(new_comb, ())
    for existing_comb in existing_combs:
        existing_comb_flat = sum(existing_comb, ())
        if new_comb_flat == existing_comb_flat[::-1]:
            return True
    return False

filtered_combinations_tuples = []
for comb in all_combinations_tuples:
    if not is_combination_mirrored(comb, filtered_combinations_tuples):
        filtered_combinations_tuples.append(comb)

mon_list = [comb for comb in filtered_combinations_tuples if len(comb) == 1]
dimer_list = [comb for comb in filtered_combinations_tuples if len(comb) == 2]
trimer_list = [comb for comb in filtered_combinations_tuples if len(comb) == 3]

def monomer_in_combination(monomer, combination):
    """Check if a monomer or its mirror is in the combination."""
    mirrored_monomer = tuple(reversed(monomer))
    return any(m == monomer or m == mirrored_monomer for m in combination)

def count_monomers(combinations, monomers):
    # Initialize counts for each monomer
    counts = {m: [] for m in monomers}

    # Iterate over combinations
    for comb in combinations:
        # Flatten the combination for processing
        flattened_comb = sum(comb, ())
        
        # Count each monomer
        for monomer in monomers:
            count = sum(monomer_in_combination(monomer, (m,)) for m in flattened_comb)
            counts[monomer].append(count)

    return counts    
    
pdb.set_trace()


mon_pc_species = jnp.array(list(map(lambda t: list(sum(t, ())), mon_list)))
dimer_pc_species = jnp.array(list(map(lambda t: list(sum(t, ())), dimer_list)))
trimer_pc_species = jnp.array(list(map(lambda t: list(sum(t, ())), trimer_list)))    
"""