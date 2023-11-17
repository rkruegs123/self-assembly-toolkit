import pdb
import numpy as onp
from copy import deepcopy

import jax.numpy as jnp
from jax_md import rigid_body, energy, space


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

monA = rb_baseline.set(point_species=jnp.array([1, vertex_species, 2])) # Shape type 0
monB = rb_baseline.set(point_species=jnp.array([3, vertex_species, 4])) # Shape type 1
monC = rb_baseline.set(point_species=jnp.array([5, vertex_species, 6])) # Shape type 2

target_shape = rigid_body.concatenate_shapes(monA, monB, monC)
n_species = len(set(onp.array(target_shape.point_species)))
target_shape_species = onp.array([0, 1, 2])









sigma_table = onp.zeros((n_species, n_species))
sigma_table[vertex_species, vertex_species] = 10000.0
sigma_table = jnp.array(sigma_table)

pair_ss_energy_fn = energy.soft_sphere_pair(
    displacement_fn,
    sigma=sigma_table,
    species=n_species
)


default_weak_eps = 1.0
eps_table = onp.full((n_species, n_species), default_weak_eps)
eps_table[vertex_species, :] = 0.0
eps_table[:, vertex_species] = 0.0
default_strong_eps = 10.0
eps_table[onp.array([2, 3, 4, 5]), onp.array([3, 2, 5, 4])] = default_strong_eps
eps_table = jnp.array(eps_table)

pair_morse_energy_fn = energy.morse_pair(
    displacement_fn,
    sigma=0.0,
    epsilon=eps_table,
    alpha=5.0,
    species=n_species
)


target_ss_energy_fn = rigid_body.point_energy(
    pair_ss_energy_fn, target_shape
)
target_morse_energy_fn = rigid_body.point_energy(
    pair_morse_energy_fn, target_shape
)
def target_energy_fn(body, **kwargs):
    return target_ss_energy_fn(body, **kwargs) + target_morse_energy_fn(body, **kwargs)





# Option 1
ot1_shape = rigid_body.concatenate_shapes(monA, monB)
ot1_ss_energy_fn = rigid_body.point_energy(
    pair_ss_energy_fn, ot1_shape
)
ot1_morse_energy_fn = rigid_body.point_energy(
    pair_morse_energy_fn, ot1_shape
)
def ot1_energy_fn(body, **kwargs):
    return ot1_ss_energy_fn(body, **kwargs) + ot1_morse_energy_fn(body, **kwargs)

ot2_shape = rigid_body.concatenate_shapes(monA, monB)
ot2_ss_energy_fn = rigid_body.point_energy(
    pair_ss_energy_fn, ot2_shape
)
ot2_morse_energy_fn = rigid_body.point_energy(
    pair_morse_energy_fn, ot2_shape
)
def ot2_energy_fn(body, **kwargs):
    return ot2_ss_energy_fn(body, **kwargs) + ot2_morse_energy_fn(body, **kwargs)

ot1_energy = ot1_energy_fn(body)
ot2_energy = ot2_energy_fn(body)


# Option 2
mon = deepcopy(rb_baseline) # has dummy point species
dimer_shape = rigid_body.concatenate_shapes(mon, mon)
ot_ss_energy_fn = rigid_body.point_energy(
    pair_ss_energy_fn, dimer_shape
)
ot_morse_energy_fn = rigid_body.point_energy(
    pair_morse_energy_fn, dimer_shape
)
def ot_energy_fn(body, **kwargs):
    return ot_ss_energy_fn(body, **kwargs) + ot_morse_energy_fn(body, **kwargs)


ot1_energy = ot_energy_fn(body, species=jnp.array([1, 0, 2, 3, 0, 4])) # evaluate for ot1
ot2_energy = ot_energy_fn(body, species=jnp.array([3, 0, 4, 5, 0, 6])) # evaluate for ot2









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

    rb_energy = target_energy_fn(target_body)
    pdb.set_trace()

    return



if __name__ == "__main__":
    run()
