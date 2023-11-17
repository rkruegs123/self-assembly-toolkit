import pdb
import numpy as onp

import jax.numpy as jnp
from jax_md import rigid_body, energy, space
from jax.config import config
config.update("jax_enable_x64", True)

displacement_fn, shift_fn = space.free()

r_vertex = 1.0
r_patch = 0.1

vertex_species = 0
rb_pos = jnp.array([
    [-(r_vertex+r_patch), 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [(r_vertex+r_patch), 0.0, 0.0]
])



# point_union_shape just takes positions of spheres and unites them together in a rb (with mass)
#comes from:def point_union_shape(points: Array, masses: Array) -> RigidPointUnion:
# RigidPointUnion(points, masses, point_count, point_offset, point_species=None, point_radius=<factory>)
#Construct a rigid body out of points and masses.
rb_baseline = rigid_body.point_union_shape(
    rb_pos, 1.0 # note: dummy_mass
)

#You can set the oint_species=None from the RigidPointUnion
monA = rb_baseline.set(point_species=jnp.array([1, vertex_species, 2])) # Shape type 0
monB = rb_baseline.set(point_species=jnp.array([3, vertex_species, 4])) # Shape type 1
monC = rb_baseline.set(point_species=jnp.array([5, vertex_species, 6])) # Shape type 2

#def concatenate_shapes(*shapes) -> RigidPointUnion:
#Concatenate a list of RigidPointUnions into a single RigidPointUnion."
target_shape = rigid_body.concatenate_shapes(monA, monB, monC)

#set() makes sure that there are no repeated elements, 
#target_shape.point_species becasue this is a version of RigidPointUnion you can take out the whole species list = [1,0,2,3,0,4,5,0,6]
n_species = len(set(onp.array(target_shape.point_species)))


target_shape_species = onp.array([0, 1, 2])  #I dont think this is being used anywhere?

#attraction
sigma_table = onp.zeros((n_species, n_species))
sigma_table[vertex_species, vertex_species] = 10000.0
#from onp array to jnp array for easier use
sigma_table = jnp.array(sigma_table)

#repulsion function
pair_ss_energy_fn = energy.soft_sphere_pair(
    displacement_fn,
    sigma=sigma_table,
    species=n_species
)


default_weak_eps = 1.0
default_strong_eps = 10.0
eps_table = onp.full((n_species, n_species), default_weak_eps)
eps_table[vertex_species, :] = 0.0
eps_table[:, vertex_species] = 0.0
eps_table[onp.array([2, 3, 4, 5]), onp.array([3, 2, 5, 4])] = default_strong_eps
eps_table = jnp.array(eps_table)

#attraction function
pair_morse_energy_fn = energy.morse_pair(
    displacement_fn,
    sigma=0.0,
    epsilon=eps_table,
    alpha=5.0,
    species=n_species
)

"""point_energy takes a pointwise energy function that computes the
  energy of a set of particle positions along with a RigidPointUnion
  (optionally with shape species information) and produces a new energy
  function that computes the energy of a collection of rigid bodies"""

target_ss_energy_fn = rigid_body.point_energy(
    pair_ss_energy_fn, target_shape
)
target_morse_energy_fn = rigid_body.point_energy(
    pair_morse_energy_fn, target_shape
)
"""Arbitrary Keyword Arguments: When you see **kwargs in a function definition, it means that the function can accept any number of keyword arguments in addition to the explicitly defined parameters.
Dictionary Format: Inside the function, kwargs is treated as a dictionary. Each keyword argument is stored as a key-value pair in this dictionary. For instance, if you call a function with my_function(arg1='value1', arg2='value2'), inside the function, kwargs would be {'arg1': 'value1', 'arg2': 'value2'}
Makes everything more flexible: It allows a function to handle a variety of arguments that you might not want to specify explicitly in the function definition. This is particularly useful in cases where the function might need to handle different options or configurations.
"""
def dimer_off_target_energy_fn(body, off_target_shape):
    # These functions should calculate the energy and return a value, not another function
    ss_energy = rigid_body.point_energy(pair_ss_energy_fn, off_target_shape)(body)
    morse_energy = rigid_body.point_energy(pair_morse_energy_fn, off_target_shape)(body)
    return ss_energy + morse_energy
    

    

def target_energy_fn(body, **kwargs):
    return target_ss_energy_fn(body, **kwargs) + target_morse_energy_fn(body, **kwargs)


def run():
    target_ref_dist = 2*(r_vertex + r_patch)
    R_target = jnp.array([
        [0.0, 0.0, 0.0],
        [target_ref_dist, 0.0, 0.0],
        [2*target_ref_dist, 0.0, 0.0]
    ])
    Q_vec_target = jnp.full((3, 4), jnp.array([1.0, 0.0, 0.0, 0.0]))
    Q_target = rigid_body.Quaternion(Q_vec_target)
    target_body = rigid_body.RigidBody(R_target, Q_target)
    
    target_energy = target_energy_fn(target_body)
    
    monomers = [monA, monB, monC] 
    
    R_off_target = jnp.array([
        [0.0, 0.0, 0.0],
        [target_ref_dist, 0.0, 0.0]
    ])
    Q_vec_off_target = jnp.full((2, 4), jnp.array([1.0, 0.0, 0.0, 0.0]))
    Q_off_target = rigid_body.Quaternion(Q_vec_target)
    off_target_body = rigid_body.RigidBody(R_target, Q_target)
    
    off_target_energies= []
    for i in range(len(monomers)):
        off_target_shape = rigid_body.concatenate_shapes(monomers[i], monomers[i])
        off_target_energy = dimer_off_target_energy_fn(off_target_body, off_target_shape)
        off_target_energies.append(off_target_energy)
    for i in range(len(monomers)):
        for j in range(len(monomers)):
            if i != j: 
                off_target_shape = rigid_body.concatenate_shapes(monomers[i], monomers[i])
                off_target_energy = dimer_off_target_energy_fn(off_target_body, off_target_shape)
                off_target_energies.append(off_target_energy)
    off_target_energies = jnp.array(off_target_energies)
    tot_off_target_energy = jnp.sum(off_target_energies)
    

    tot_energy = tot_off_target_energy + target_energy
    
    #pdb.set_trace()

    return tot_energy



if __name__ == "__main__":
    print(run())
