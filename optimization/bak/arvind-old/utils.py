import numpy as onp

import jax.numpy as jnp

from transformations import transformations as jts


# euler_scheme: string of 4 characters (e.g. 'sxyz') that define euler angles
euler_scheme = "sxyz"

def convert_to_matrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T, R)
