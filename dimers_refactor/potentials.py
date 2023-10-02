import jax.numpy as jnp


def smoothing(r, ron, rcut):
    if r < ron:
        return 1
    if r > rcut:
        return 0
    return ((rcut**2-r**2)**2 * (rcut**2+2*r**2-3*ron**2) ) / ( (rcut**2 - ron**2)**3)


def morse(r, rmin, rmax, D0, alpha, r0):
    if r>= rmax:
        return 0.
    return D0*(jnp.exp(-2*alpha*(r-r0)) - 2*jnp.exp(-alpha*(r-r0)))


def morse_x(r, rmin, rmax, D0, alpha, r0, ron):
    return morse(r, rmin, rmax, D0, alpha, r0)*smoothing(r, ron, rmax)


def morse_x_repulsive(r, rmin, rmax, D0, alpha, r0, ron):
    # FIXME: isn't this just -morse_x(r, rmin, rmax, D0, alpha, r0, ron)
    return -morse(r, rmin, rmax, D0, alpha, r0)*smoothing(r, ron, rmax)

def repulsive(r, rmin, rmax, A, alpha):
    if r >= rmax:
        return 0.
    return (A/(alpha*rmax)) * (rmax-r)**alpha
