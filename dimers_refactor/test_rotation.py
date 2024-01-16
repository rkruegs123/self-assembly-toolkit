import pdb
from tqdm import tqdm
import argparse
import numpy as onp
import scipy as osp
import csv
import matplotlib.pyplot as plt
import time
from random import randint
from jax import lax
from functools import partial

from jax import random
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
import jax.numpy as jnp
import optax

import potentials
from jax_transformations3d import jax_transformations3d as jts
from utils import euler_scheme, convert_to_matrix, ref_ppos, setup_ref_q0


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
  masked = jnp.where(mask, operand, 0)
  return jnp.where(mask, fn(masked), placeholder)

def distance(dR):
  dr = jnp.sum(dR ** 2, axis=-1)
  return safe_mask(dr > 0, jnp.sqrt, dr)


dist_fn = distance



some_big_number = 100
factorial_table = jnp.array([osp.special.factorial(x) for x in range(some_big_number)])
comb_table = onp.zeros((some_big_number, some_big_number))
for i in range(some_big_number):
    for j in range(some_big_number):
        if i >= j:
            comb_table[i, j] = osp.special.comb(i, j)
comb_table = jnp.array(comb_table)




def get_energy_fns(args):

    Nbb = 2

    def monomer_energy(q, ppos):

        return jnp.float64(0)


    sphere_radius = 1.0
    patch_radius = 0.2 * sphere_radius


    morse_rcut = 8. / args['morse_a'] + args['morse_r0']
    def cluster_energy(q, ppos):
     
        Mat = []
        for i in range(Nbb):
            qi = i*6
            Mat.append(convert_to_matrix(q[qi:qi+6]))


        real_ppos = []
        for i in range(Nbb):
            real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

        tot_energy = jnp.float64(0)

        # Add repulsive interaction between spheres
        def j_repulsive_fn(j, pos1):
            pos2 = real_ppos[1][j]
            # r = jnp.linalg.norm(pos1-pos2)
            r = dist_fn(pos1-pos2)
            return potentials.repulsive(
                r, rmin=0, rmax=sphere_radius*2,
                A=args['rep_A'], alpha=args['rep_alpha'])

        def i_repulsive_fn(i):
            pos1 = real_ppos[0][i]
            all_j_terms = vmap(j_repulsive_fn, (0, None))(jnp.arange(3), pos1)
            return jnp.sum(all_j_terms)

        repulsive_sm = jnp.sum(vmap(i_repulsive_fn)(jnp.arange(3)))
        tot_energy += repulsive_sm        

        
        # Add attraction b/w blue patches
        pos1 = real_ppos[0][3]
        pos2 = real_ppos[1][3]
        # r = jnp.linalg.norm(pos1-pos2)
        r = dist_fn(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_b'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w green patches
        pos1 = real_ppos[0][5]
        pos2 = real_ppos[1][4]
        # r = jnp.linalg.norm(pos1-pos2)
        r = dist_fn(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_g'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w red patches
        pos1 = real_ppos[0][4]
        pos2 = real_ppos[1][5]
        # r = jnp.linalg.norm(pos1-pos2)
        r = dist_fn(pos1-pos2)
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
    # assert(Nbb*6 == q0.shape[0])
    # assert(len(ppos.shape) == 3)
    # assert(ppos.shape[0] == Nbb)
    # assert(ppos.shape[2] == 3)

    E = energy_fn(q0, ppos)
    G = grad(energy_fn)(q0, ppos)
    H = hessian(energy_fn)(q0, ppos)

    evals, evecs = jnp.linalg.eigh(H)

    # print("\nEval", evals)

    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(evals < zeromode_thresh, 1, 0))

    if Nbb == 1:
        zvib = 1.0
    else:
        zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))

    # print("Zvib", zvib)

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

def standard_error(data):
    mean = jnp.mean(data, axis=0)

    # Calculate the standard error using the formula: std(data) / sqrt(N)
    std_dev = jnp.std(data, axis=0)
    sqrt_n = jnp.sqrt(data.shape[0])
    std_error = std_dev / sqrt_n

    return std_error


def calc_jmean(f, key, nrandom=100000):
    def random_euler_angles(key):
        quat = jts.random_quaternion(None, key)
        return jnp.array(jts.euler_from_quaternion(quat, euler_scheme))

    def set_nu(angles):
        nu0 = jnp.full((12,), 0.0)
        return nu0.at[3:6].set(angles)

    def set_nu_random(key):
        return set_nu(random_euler_angles(key))

    key, *splits = random.split(key, nrandom+1)
    nus = vmap(set_nu_random)(jnp.array(splits))

    nu_fn = jit(lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu, False))))
    Js = vmap(nu_fn)(nus)
    # pdb.set_trace()
    mean = jnp.mean(Js)
    # error = osp.stats.sem(Js)
    # error = osp.stats.sem(Js)
    error = standard_error(Js)

    return mean, error


def calculate_zc(key, energy_fn, all_q0, all_ppos, sigma, kBT, V):

    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, all_q0, all_ppos)

    Js_mean, Js_error = calc_jmean(f, key)
    Jtilde = 8.0*(jnp.pi**2) * Js_mean

    #E0 = energy_fn(all_q0, all_ppos)
    #boltzmann_weight = jnp.exp(-E0/kBT)

    #return boltzmann_weight * V * (Jtilde/sigma) * zvib
    return Jtilde



N_mon_real = 9
def Calculate_pc_list(N_mon, Zc_monomer, Zc_dimer, exact=False):
    # nd_fact = jax_factorial(N_mon_real)

    def Mc(Nd):
        return comb_table[N_mon_real, Nd] * comb_table[N_mon_real, Nd] * factorial_table[Nd]

    def Pc(Nd):
        return Mc(Nd) * (Zc_dimer**Nd) * (Zc_monomer**(N_mon_real-Nd)) * (Zc_monomer**(N_mon_real-Nd))

    pc_list = vmap(Pc)(jnp.arange(N_mon_real+1))
    return pc_list / jnp.sum(pc_list)





def Calculate_yield_can(Nb_dummy, Nr_dummy, pc_list):
    Nb = 9
    Nr = 9

    Y_list = vmap(lambda Nd: Nd / (Nb+Nr-Nd))(jnp.arange(N_mon_real+1))
    return jnp.dot(Y_list, pc_list)


def run(args, noise_terms, seed=0):

    key = random.PRNGKey(seed)

    monomer_energy, dimer_energy = get_energy_fns(args)

    Nblue, Nred = args['num_monomer'], args['num_monomer']

    conc = args['conc']
    Ntot = jnp.sum(jnp.array(args['num_monomer']))
    V = Ntot / conc
    ref_q0 = setup_ref_q0(noise_terms)
    split1, split2 = random.split(key)
    Zc_dimer = calculate_zc(
        split1, dimer_energy, ref_q0, ref_ppos,
        sigma=1, kBT=1.0, V=V)

    return Zc_dimer

    

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

    
def main():
    parser = get_argparse()
    args = vars(parser.parse_args())  

  
    noise_terms = 1e-15  
    seed = 0 

   
    JTilde = run(args, noise_terms, seed)


    print("JTilde:", JTilde)

if __name__ == "__main__":
    main()

