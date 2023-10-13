import pdb
from tqdm import tqdm
import argparse
import numpy as np
import scipy as osp
import csv
import matplotlib.pyplot as plt
import time
from random import randint

from jax import random
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
import jax.numpy as jnp
from jax.ops import index, index_add, index_update

import potentials
from jax_transformations3d import jax_transformations3d as jts
from utils import euler_scheme, convert_to_matrix, ref_ppos, ref_q0

from jax.config import config
config.update("jax_enable_x64", True)


def get_energy_fns(args):

    Nbb = 2

    def monomer_energy(q, ppos):
        # assert(q.shape[0] == 6)
        return jnp.float64(0)


    sphere_radius = 1.0
    patch_radius = 0.2 * sphere_radius
    # types: ['A', 'B1', 'B2', 'G1', 'G2', 'R1', 'R2']
    # BBt[0].typeids: array([0, 0, 0, 1, 5, 3])
    # BBt[1].typeids: array([0, 0, 0, 2, 4, 6])

    morse_rcut = 8. / args['morse_a'] + args['morse_r0']
    def cluster_energy(q, ppos):
        # convert the building block coordinates to a tranformation
        # matrix
        Mat = []
        for i in range(Nbb):
            qi = i*6
            Mat.append(convert_to_matrix(q[qi:qi+6]))

        # apply building block matrix to spheres positions
        real_ppos = []
        for i in range(Nbb):
            real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

        tot_energy = jnp.float64(0)

        # Add repulsive interaction between spheres
        for i in range(3):
            pos1 = real_ppos[0][i]
            for j in range(3):
                pos2 = real_ppos[1][j]
                r = jnp.linalg.norm(pos1-pos2)
                tot_energy += potentials.repulsive(
                    r, rmin=0, rmax=sphere_radius*2,
                    A=args['rep_A'], alpha=args['rep_alpha'])

        # Add attraction b/w blue patches
        pos1 = real_ppos[0][3]
        pos2 = real_ppos[1][3]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_b'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w green patches
        pos1 = real_ppos[0][5]
        pos2 = real_ppos[1][4]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_g'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w red patches
        pos1 = real_ppos[0][4]
        pos2 = real_ppos[1][5]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_r'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Note: no repulsion between identical patches, as in Agnese's code. May affect simulations.
        return tot_energy

    return monomer_energy, cluster_energy

def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb,Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans, angles))

def add_variables_all(mas, mbs):
    """
    Given two vectors of length (6*n,), 'add' them per building block according
    to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))

    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(
        mas_temp, mbs_temp), mas.shape)


def setup_variable_transformation(energy_fn, q0, ppos):
    """
    Args:
    energy_fn: function to calculate the energy:
    E = energy_fn(q, euler_scheme, ppos)
    q0: initial coordinates (positions and orientations) of the building blocks
    ppos: "patch positions", array of shape (N_bb, N_patches, dimension)

    Returns: function f that defines the coordinate transformation, as well as
    the number of zero modes (which should be 6) and Z_vib

    Note: we assume without checking that ppos is defined such that all the euler
    angels in q0 are initially 0.
    """

    Nbb = q0.shape[0] // 6 # Number of building blocks
    assert(Nbb*6 == q0.shape[0])
    assert(len(ppos.shape) == 3)
    assert(ppos.shape[0] == Nbb)
    assert(ppos.shape[2] == 3)

    E = energy_fn(q0, ppos)
    G = grad(energy_fn)(q0, ppos)
    H = hessian(energy_fn)(q0, ppos)

    evals, evecs = jnp.linalg.eigh(H)

    print("\nEval", evals)

    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(evals < zeromode_thresh, 1, 0))

    if Nbb == 1:
        zvib = 1.0
    else:
        zvib = jnp.product(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))

    print("Zvib", zvib)

    def ftilde(nu):
        return jnp.matmul(evecs.T[6:].T, nu[6:])

    def f_multimer(nu, addq0=True):
        # q0+ftilde
        dq_tilde = ftilde(nu)

        q_tilde = jnp.where(addq0, add_variables_all(q0, dq_tilde), dq_tilde)

        nu_bar_repeat = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return add_variables_all(q_tilde, nu_bar_repeat)

    def f_monomer(nu, addq0=True):
        return nu

    if Nbb == 1:
        f = f_monomer
    else:
        f = f_multimer

    return jit(f), num_zero_modes, zvib


def calc_jmean(f, key, nrandom=100000):
    def random_euler_angles(key):
        quat = jts.random_quaternion(None, key)
        return jnp.array(jts.euler_from_quaternion(quat, euler_scheme))

    def set_nu(angles):
        nu0 = jnp.full((12,), 0.0)
        return index_update(nu0, index[3:6], angles)

    def set_nu_random(key):
        return set_nu(random_euler_angles(key))

    key, *splits = random.split(key, nrandom+1)
    nus = vmap(set_nu_random)(jnp.array(splits))

    nu_fn = jit(lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu, False))))
    Js = vmap(nu_fn)(nus)

    mean = jnp.mean(Js)
    error = osp.stats.sem(Js)

    return mean, error


def calculate_zc(key, energy_fn, all_q0, all_ppos, sigma, kBT, V):

    """
    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, all_q0, all_ppos)

    Js_mean, Js_error = calc_jmean(f, key)
    Jtilde = 8.0*(jnp.pi**2) * Js_mean

    E0 = energy_fn(all_q0, all_ppos)
    boltzmann_weight = jnp.exp(-E0/kBT)

    print("E0", len(all_q0), E0)
    print("zvib", len(all_q0), zvib)
    print("Jtilde", len(all_q0), Jtilde)

    return boltzmann_weight * V * (Jtilde/sigma) * zvib
    """

    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, all_q0, all_ppos)

    E0 = energy_fn(all_q0, all_ppos)
    boltzmann_weight = jnp.exp(-E0/kBT)

    return boltzmann_weight * V * zvib


def Calculate_pc_list(Nb, Nr, Zc_monomer, Zc_dimer, exact=False):
    Nd_max = min(Nb, Nr)
    def Mc(Nd):
        return osp.special.comb(Nb, Nd, exact=exact) \
            * osp.special.comb(Nr, Nd, exact=exact) \
            * osp.special.factorial(Nd, exact=exact)

    def Pc(Nd):
        return Mc(Nd) * (Zc_dimer**Nd) * (Zc_monomer**(Nb-Nd)) * (Zc_monomer**(Nr-Nd))

    pc_list = jnp.array([Pc(Nd) for Nd in range(0, Nd_max+1)])
    pc_list = pc_list / jnp.sum(pc_list)

    return pc_list


def Calculate_yield_can(Nb, Nr, pc_list):
    Y_list = jnp.array([Nd / (Nb+Nr-Nd) for Nd in range(len(pc_list))])
    return jnp.dot(Y_list, pc_list)

def run(args, seed=0):
    """
    monomer_energy is a function of q=(x,y,z,alpha,beta,gamma) with the parameters "euler_scheme" and "ppos"
    dimer_energy is a function of q=(x,y,z,alpha,beta,gamma) with the parameters "euler_scheme" and "ppos"
    setup_system has to return a list, corresponding to all the different structures. Each element of the list is a list itself that must contain:
    - energy of the structure as a function of the 6n variables where 6 is the x,y,z,alpha,beta,gamma and n is the number of bb in the structure
    - ref_ppos: the positions of the spheres of a building block wrt the bb COM
    - the reference q0: 6-dimensional COM coordinates of the BB in the structure
    - sigma is always 1 for asymmetric structures
    """

    key = random.PRNGKey(seed)

    monomer_energy, dimer_energy = get_energy_fns(args)

    Nblue, Nred = args['num_monomer'], args['num_monomer']

    conc = args['conc']
    Ntot = jnp.sum(jnp.array(args['num_monomer']))
    V = Ntot / conc

    split1, split2 = random.split(key)

    Zc_dimer = calculate_zc(
        split1, dimer_energy, ref_q0, ref_ppos,
        sigma=1, kBT=1.0, V=V)

    Zc_monomer = calculate_zc(
        split2, monomer_energy,
        ref_q0[:6], jnp.array([ref_ppos[0]]),
        sigma=1, kBT=1.0, V=V)

    pc_list = Calculate_pc_list(Nblue, Nred, Zc_monomer, Zc_dimer, exact=True)
    Y_dimer = Calculate_yield_can(Nblue, Nred, pc_list)

    # return Zc_dimer
    return Y_dimer

def get_argparse():
    parser = argparse.ArgumentParser(description='Compute the yield of a simple dimer system')

    # System setup
    parser.add_argument('-c', '--conc', type=float,  default=0.001, help='Monomer concentration')
    parser.add_argument('-n', '--num-monomer', type=int,  default=9,
                        help='Number of each kind of monomer')

    # Repulsive interaction
    parser.add_argument('--rep-A', type=float,  default=500.0,
                        help='A parameter for repulsive interaction')
    parser.add_argument('--rep-alpha', type=float,  default=2.5,
                        help='alpha parameter for repulsive interaction')

    # Morse interaction
    parser.add_argument('--morse-d0', type=float,  default=10.0,
                        help='d0 parameter for Morse interaction')
    parser.add_argument('--morse-d0-r', type=float,  default=1.0,
                        help='Scalar for d0 for red patches')
    parser.add_argument('--morse-d0-g', type=float,  default=1.0,
                        help='Scalar for d0 for green patches')
    parser.add_argument('--morse-d0-b', type=float,  default=1.0,
                        help='Scalar for d0 for blue patches')

    parser.add_argument('--morse-a', type=float,  default=5.0,
                        help='alpha parameter for Morse interaction')
    parser.add_argument('--morse-r0', type=float,  default=0.0,
                        help='r0 parameter for Morse interaction')

    return parser


target_yield = 0.5
def loss_fn(d0, args):
    args['morse_d0'] = d0
    ys = run(args)
    return (target_yield - ys)**2
grad_fn = jacfwd(loss_fn)


if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())


    pdb.set_trace()

    my_grad = grad_fn(4.0, args)

    pdb.set_trace()

    print("done")
