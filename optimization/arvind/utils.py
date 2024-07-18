import jax.numpy as jnp
from jax import vmap
from jax_transformations3d import jax_transformations3d as jts

euler_scheme = "sxyz"

def convert_to_matrix(mi):
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T, R)

def get_positions(q, ppos):
    Mat = []
    for i in range(len(ppos)):
        qi = i * 6
        Mat.append(convert_to_matrix(q[qi:qi+6]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(jts.matrix_apply(mat, ppos[i]))

    real_ppos = jnp.array(real_ppos)
    real_ppos = real_ppos.reshape(-1, 3)

    return real_ppos

def add_variables(ma, mb):
    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb, Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))
    return jnp.concatenate((trans, angles))

def add_variables_all(mas, mbs):
    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))
    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(mas_temp, mbs_temp), mas.shape)




