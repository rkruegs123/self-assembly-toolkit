import jax.numpy as jnp
from jax import jit


@jit
def Potential_S(r, ron, rcut):
    return jnp.where(
        r < ron, 1.0,
        jnp.where(r > rcut, 0.0,
                  ((rcut**2-r**2)**2 * (rcut**2+2*r**2-3*ron**2)) / ((rcut**2 - ron**2)**3)))

@jit
def morse_E(r, rmin, rmax, D0, alpha, r0):
    # D0, alpha, r0, rmax = rmin,rmax,D0,alpha,r0
    return jnp.where(r >= rmax, 0.0,
                     D0*(jnp.exp(-2*alpha*(r-r0)) - 2*jnp.exp(-alpha*(r-r0))))

@jit
def morseX_E(r, rmin, rmax, D0, alpha, r0, ron):
    return morse_E(r, rmin, rmax, D0, alpha, r0) * Potential_S(r, ron, rmax)
