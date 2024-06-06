import pdb
import numpy as onp
import jax.numpy as jnp
from copy import deepcopy
from jax import vmap
from jax_md import rigid_body, energy, util, space, dataclasses
from jax_md.rigid_body import RigidPointUnion, union_to_points
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)
euler_scheme = "sxyz"

displacement_fn, shift_fn = space.free()


r_vertex = 1.0
r_patch = 0.1

vertex_species = 0
rb_pos = jnp.array([
    [-(r_vertex+r_patch), 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [(r_vertex+r_patch), 0.0, 0.0]
])
rb_baseline = rigid_body.point_union_shape(
    rb_pos, 1.0 # note: dummy_mass
)

n_species = 7


small_value = 1e-17  # Small value to replace zeros to avoid nans

sigma_table = onp.full((n_species, n_species), small_value)  
sigma_table[vertex_species, vertex_species] = 1000.0  
sigma_table = jnp.array(sigma_table)

default_weak_eps = 0.5
eps_table = onp.full((n_species, n_species), default_weak_eps)
eps_table[vertex_species, :] = 0.0
eps_table[:, vertex_species] = 0.0
default_strong_eps = 1.0
eps_table[onp.array([2, 3, 4, 5]), onp.array([3, 2, 5, 4])] = default_strong_eps
eps_table = jnp.array(eps_table)




pair_ss_energy_fn = energy.soft_sphere_pair(
    displacement_fn,
    sigma=sigma_table,
    species=n_species
)


pair_morse_energy_fn = energy.morse_pair(
    displacement_fn,
    sigma=0.0,
    epsilon=eps_table,
    alpha=2.0,
    species=n_species
)

mon_dist = 2*(r_vertex + r_patch)

R_2 = jnp.array([
    [0.0, 0.0, 0.0],
    [mon_dist, 0.0, 0.0],
])

Q_vec_2 = jnp.full((2, 4), jnp.array([1.0, 0.0, 0.0, 0.0]))
Q_quat_2 = rigid_body.Quaternion(Q_vec_2)
body_2 = rigid_body.RigidBody(R_2, Q_quat_2)

R_3 = jnp.array([
    [0.0, 0.0, 0.0],
    [mon_dist, 0.0, 0.0],
    [2*mon_dist, 0.0, 0.0]
])

Q_vec_3 = jnp.full((3, 4), jnp.array([1.0, 0.0, 0.0, 0.0]))
Q_quat_3 = rigid_body.Quaternion(Q_vec_3)
body_3 = rigid_body.RigidBody(R_3, Q_quat_3)

pos_2, _ = rigid_body.union_to_points(body_2, rb_baseline, shape_species= None )
pos_3, _ = rigid_body.union_to_points(body_3, rb_baseline, shape_species= None )

print(pos_3) 

pos_2_flat = pos_2.flatten()
pos_3_flat = pos_3.flatten()
#print(pos_3_flat) 
#pdb.set_trace()

def rb_energy_fn(pos_flat, point_species,  **kwargs):
    num_particles = len(pos_flat) // 3 
    # Reshape the pos_flat array back to its original shape 
    pos = pos_flat.reshape(num_particles, 3)
    #pdb.set_trace()
    if point_species is None:
        return pair_morse_energy_fn(pos, **kwargs) + pair_ss_energy_fn(pos,  **kwargs)
    return pair_morse_energy_fn(pos, species=point_species, **kwargs) + pair_ss_energy_fn(pos, species=point_species, **kwargs)


def hess(energy_fn, pos_flat, species):

    H = hessian(energy_fn)(pos_flat, point_species=species)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs

    """
def get_zvib(energy_fn, pos, species):
    evals, evecs = hess(energy_fn, pos, species)
    zeromode_thresh = 1e-8
    zvib = jnp.prod(jnp.sqrt(2. * jnp.pi / (jnp.abs(evals[9:]) + 1e-12)))
    """

def get_zvib(energy_fn, pos, species):
    evals, evecs = hess(energy_fn, pos, species)
    pdb.set_trace()
    zero_mode_threshold = 1e-8
    relevant_evals = evals[evals > zero_mode_threshold]
    zvib = jnp.prod(jnp.sqrt(2. * jnp.pi / (jnp.abs(relevant_evals) + 1e-12)))
  
    
    return zvib


# Dimer
ot1_species = onp.array([1, 0, 2, 3, 0, 4])
energy_ot = rb_energy_fn(pos_2_flat, ot1_species)
dimer_zvib = get_zvib(rb_energy_fn, pos_2_flat,  ot1_species)




pdb.set_trace()


ot1_species = onp.array([1, 0, 2, 3, 0, 4])
target_species = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6])

energy_ot = rb_energy_fn(pos_2_flat, ot1_species)
energy_target = rb_energy_fn(pos_3_flat, target_species)
print(energy_ot )
print(energy_target )

print(get_zvib(rb_energy_fn, pos_2_flat,  ot1_species))
print(get_zvib(rb_energy_fn, pos_3_flat,  target_species))

pdb.set_trace()
