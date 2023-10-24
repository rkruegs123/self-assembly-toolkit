import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)


def smoothing(r, ron, rcut):
    if r < ron:
        return 1
    if r > rcut:
        return 0
    return ((rcut**2-r**2)**2 * (rcut**2+2*r**2-3*ron**2) ) / ( (rcut**2 - ron**2)**3)
    #return jnp.where(r < ron, 1, jnp.where(r > rcut, 0, ((rcut**2 - r**2)**2 * (rcut**2 + 2*r**2 - 3*ron**2)) / ((rcut**2 - ron**2)**3)))


def morse(r, rmin, rmax, D0, alpha, r0):
    return jnp.where(r >= rmax, 0.0, D0*(jnp.exp(-2*alpha*(r-r0)) - 2*jnp.exp(-alpha*(r-r0))))

def morse_x(r, rmin, rmax, D0, alpha, r0, ron):
    #return morse(r, rmin, rmax, D0, alpha, r0)*smoothing(r, ron, rmax)
    return morse(r, rmin, rmax, D0, alpha, r0)

def morse_x_repulsive(r, rmin, rmax, D0, alpha, r0, ron):
    # FIXME: isn't this just -morse_x(r, rmin, rmax, D0, alpha, r0, ron)
    return -morse(r, rmin, rmax, D0, alpha, r0)*smoothing(r, ron, rmax)

def repulsive(r, rmin, rmax, A, alpha):
     #Option 1:
    #if r >= rmax:
        #return 0.
    #return (A/(alpha*rmax)) * (rmax-r)**alpha
    base = jnp.maximum(rmax - r, 0.0)
    
    # Option 2:
    #return jnp.where(r >= rmax, 0.0, (A/(alpha*rmax)) * (rmax-r)**alpha)
    return jnp.where(r < rmax, (A/(alpha*rmax)) * base**alpha, 0.0)

