import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import numpy as onp

import jax.numpy as jnp
from jax_md import rigid_body, energy, util, space, dataclasses
import optax
from jax import jit, grad, value_and_grad

import potentials
from jax_transformations3d import jax_transformations3d as jts

from jax.config import config
config.update("jax_enable_x64", True)


"""
Notes:
- we will consider only the dimer system
"""




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
num_building_blocks = 2


a = 1 # distance of the center of the spheres from the BB COM
b = .3 # distance of the center of the patches from the BB COM
shape1 = onp.array([
    [0., 0., a], # first sphere
    [0., a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # second sphere
    [0., -a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # third sphere
    [a, 0., b], # first patch
    [a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], # second patch
    [a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)]  # third patch
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


## Function to convert to ground state positions

def get_positions(q, ppos):
    Mat = []
    for i in range(num_building_blocks):
        qi = i*6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i in range(num_building_blocks):
        real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

    return real_ppos

# Visualize via injavis
points = get_positions(rb_info, shapes)
points = onp.array(points).reshape(-1, 3)
point_types = ["V", "V", "V", "P1", "P2", "P3", "V", "V", "V", "P1", "P3", "P2"]

fpath = "ground_state.pos"
box_size = 10.0
vertex_radius = a
patch_radius = 0.2*a
with open(fpath, "a") as of:
    box_line = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}\n"
    of.write(box_line)

    of.write(f"def V \"sphere {vertex_radius*2} 1c1c1c\"\n")
    of.write(f"def P1 \"sphere {patch_radius*2} 4fb06d\"\n")
    of.write(f"def P2 \"sphere {patch_radius*2} 43a5be\"\n")
    of.write(f"def P3 \"sphere {patch_radius*2} ff0000\"\n")

    for pos, pos_type in zip(points, point_types):
        pos_line = f"{pos_type} {pos[0]} {pos[1]} {pos[2]}\n"
        of.write(pos_line)
    of.write("eof\n")


fpath = "monomer.pos"
with open(fpath, "a") as of:
    box_line = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}\n"
    of.write(box_line)

    of.write(f"def V \"sphere {vertex_radius*2} 1c1c1c\"\n")
    of.write(f"def P1 \"sphere {patch_radius*2} 4fb06d\"\n")
    of.write(f"def P2 \"sphere {patch_radius*2} 43a5be\"\n")
    of.write(f"def P3 \"sphere {patch_radius*2} ff0000\"\n")

    for pos, pos_type in zip(points[:6], point_types[:6]):
        pos_line = f"{pos_type} {pos[0]} {pos[1]} {pos[2]}\n"
        of.write(pos_line)
    of.write("eof\n")


# Plot components of energy function

## Morse
show_morse = False
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
if show_morse:
    plt.plot(test_rs, morsex_energies)
    plt.axvline(x=morse_rcut/2, linestyle="--", color="red")
    plt.show()
    plt.clf()


## Soft sphere
show_ss = False

test_rs_ss = onp.linspace(0.0, 3.0, 100)
rep_A = 500.0
rep_alpha = 3.0
ss_energies = potentials.repulsive(test_rs_ss, rmin=0, rmax=vertex_radius*2, A=rep_A, alpha=rep_alpha)
if show_ss:
    plt.plot(test_rs_ss, ss_energies)
    plt.show()
    plt.clf()




# JAX-MD Energy Functions
displacement_fn, shift_fn = space.free()

# Morse

morse_jax_energies = energy.morse(test_rs_morse, sigma=0.0, epsilon=morse_d0, alpha=morse_a)
plot_compare_morse = False
if plot_compare_morse:
    plt.plot(test_rs_morse, morse_jax_energies, label="JAX-MD")
    plt.plot(test_rs_morse, morsex_energies, label="Agnese")
    plt.legend()
    plt.show()
    plt.clf()


# Soft sphere

# Note: units are different. Might as well do grad. descent to find the matching parameters
lr = 0.1
optimizer = optax.adam(learning_rate=lr)
params = jnp.array([rep_A, rep_alpha])
opt_state = optimizer.init(params)

opt_rs = jnp.linspace(0, vertex_radius*2, 100)
def loss_fn(params):
    ss_jax_energies = energy.soft_sphere(opt_rs, sigma=vertex_radius*2,
                                         epsilon=params[0], alpha=params[1])
    rmse = jnp.mean(jnp.sqrt((ss_jax_energies - ss_energies)**2))
    return rmse
grad_fn = jit(value_and_grad(loss_fn))

num_iters = 100000
for i in tqdm(range(num_iters)):

    loss, grads = grad_fn(params)
    if i % 100 == 0:
        print(f"Iter {i}: {loss}")
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)


rep_a_jax = params[0]
rep_alpha_jax = params[1]
pdb.set_trace()
ss_jax_energies = energy.soft_sphere(test_rs_ss, sigma=vertex_radius*2,
                                     epsilon=rep_a_jax, alpha=rep_alpha_jax)
plt.plot(test_rs_ss, ss_jax_energies, label="JAX-MD")
plt.plot(test_rs_ss, ss_energies, label="Agnese")
plt.legend()
plt.show()








# s.pose we have a pairwise function `repulsion(dr, sigma, a, alpha)`
# sigma_table, a_table, alpha_table
# a species for each particle

# also we have a bunch of postiions, `positions`

def pairwise_repulsion(i, j):
    ipos = positions[i]
    jpos = positions[j]

    i_species = species[i]
    j_species = species[j]

    sigma = sigma_table[i_species, j_species]
    alpha = alpha_table[i_species, j_species]
    a = a_table[i_species, j_species]

    dr = space.distance(displacement_fn(ipos, jpos))

    return jnp.where(i == j, 0.0, repulsion(dr, sigma, a, alpha))
total_repulsion_fn = vmap(pairwise_repulsion, (0, 0))
total_repulsion_val = total_repulsion_fn(jnp.arange(n), jnp.arange(n))
