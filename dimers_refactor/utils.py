import numpy as onp

import jax.numpy as jnp

from jax_transformations3d import jax_transformations3d as jts


# euler_scheme: string of 4 characters (e.g. 'sxyz') that define euler angles
euler_scheme = "sxyz"

def convert_to_matrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T,R)


a = 1 # distance of the center of the spheres from the BB COM
b = .3 # distance of the center of the patches from the BB COM
ref_ppos1 = onp.array([
    [0., 0., a], # first sphere
    [0., a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # second sphere
    [0., -a*onp.cos(onp.pi/6.), -a*onp.sin(onp.pi/6.)], # third sphere
    [a, 0., b], # first patch
    [a, b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)], # second patch
    [a, -b*onp.cos(onp.pi/6.), -b*onp.sin(onp.pi/6.)]  # third patch
])

# these are the positions of the spheres within the building block
ref_ppos2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                   jnp.array([1, 0, 0], dtype=jnp.float64)),
                             ref_ppos1
                         )
ref_ppos2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0, 0, 0], dtype=jnp.float64),
                                                   jnp.array([0, 1, 0], dtype=jnp.float64)),
                             ref_ppos2
)
ref_ppos = jnp.array([ref_ppos1, ref_ppos2])

separation = 2.
noise = 1e-15
# noise = 1e-16
ref_q0 = jnp.array([-separation/2.0, noise, 0, 0, 0, 0,
                    separation/2.0, 0, 0, 0, 0, 0], dtype=jnp.float64)

