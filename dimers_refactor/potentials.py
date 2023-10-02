import jax.numpy as jnp


def Potential_S(r,ron,rcut):
    if r < ron:
        return 1
    if r > rcut:
        return 0
    return ((rcut**2-r**2)**2 * (rcut**2+2*r**2-3*ron**2) ) / ( (rcut**2 - ron**2)**3)


class MorsePotential():
    #parameters:
    # rmin --> assumed to be zero!
    # rmax
    # D0
    # alpha
    # r0

    @staticmethod
    def GetDefaultParams():
        return dict(D0=0, alpha=5.0, r0=1)

    @staticmethod
    def E(r, rmin, rmax, D0, alpha, r0):
        # D0, alpha, r0, rmax = rmin,rmax,D0,alpha,r0
        if r>= rmax:
            return 0.
        return D0*(jnp.exp(-2*alpha*(r-r0)) - 2*jnp.exp(-alpha*(r-r0)))





class MorseXPotential(MorsePotential):
    # parameters:
    # rmin --> assumed to be zero!
    # rmax
    # D0
    # alpha
    # r0
    # ron

    @staticmethod
    def GetDefaultParams():
        return dict(D0=0, alpha=5.0, r0=1, ron=0.9)

    @staticmethod
    def E(r, rmin, rmax, D0, alpha, r0, ron):
        return MorsePotential().E(r, rmin, rmax, D0, alpha, r0)*Potential_S(r, ron, rmax)




class MorseXRepulsivePotential(MorsePotential):
    # parameters:
    # rmin --> assumed to be zero!
    # rmax
    # D0
    # alpha
    # r0
    # ron

    @staticmethod
    def GetDefaultParams():
        return dict(D0=0, alpha=5.0, r0=1, ron=0.9)

    @staticmethod
    def E(r, rmin, rmax, D0, alpha, r0, ron):
        return -MorsePotential().E(r, rmin, rmax, D0, alpha, r0)*Potential_S(r, ron, rmax)


class RepulsivePotential():
    # parameters:
    # rmin
    # rmax
    # A
    # alpha

    @staticmethod
    def GetDefaultParams():
        return dict(A=0, alpha=2.5)

    @staticmethod
    def E(r, rmin, rmax, A, alpha):
        if r >= rmax:
            return 0.
        return (A/(alpha*rmax)) * (rmax-r)**alpha
