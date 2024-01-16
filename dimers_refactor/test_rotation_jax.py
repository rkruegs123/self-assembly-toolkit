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
from jax_md import rigid_body, energy, util, space, dataclasses

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
    sphere_radius = 1.0

    morse_alpha = args['morse_a']
    morse_r_onset = 8. / args['morse_a'] / 2 + args['morse_r0']  # Half of morse_rcut
    morse_r_cutoff = 8. / args['morse_a'] + args['morse_r0']


    soft_sphere_sigma = 1.0
    soft_sphere_epsilon = args['rep_A']
    soft_sphere_alpha = args['rep_alpha']

    displacement_fn, shift_fn = space.free()


    soft_sphere_energy_fn = energy.soft_sphere_pair(displacement_fn,
                                                    sigma=soft_sphere_sigma,
                                                    epsilon=soft_sphere_epsilon,
                                                    alpha=soft_sphere_alpha)


    def monomer_energy(q, ppos):
        
        return jnp.float64(0)

    def cluster_energy(q, ppos):
        total_energy = jnp.float64(0)
        Mat = [convert_to_matrix(q[i*6:i*6+6]) for i in range(Nbb)]
        real_ppos = [jts.matrix_apply(Mat[i], ppos[i]) for i in range(Nbb)]

        all_pos = jnp.concatenate(real_ppos, axis=0)  
        total_energy += jnp.sum(soft_sphere_energy_fn(all_pos))
        
        def calculate_morse_energy(pos1, pos2, epsilon):
            morse_energy_fn = energy.morse_pair(
                displacement_fn,
                sigma=1.0,
                epsilon=epsilon,
                alpha=morse_alpha,
                r_onset=morse_r_onset,
                r_cutoff=morse_r_cutoff
            )
            pos_pair = jnp.vstack([pos1, pos2])
            return morse_energy_fn(pos_pair)

        for color in ['b', 'g', 'r']:
            pos1_index, pos2_index, d0_key = {
                'b': (3, 3, 'morse_d0_b'),
                'g': (5, 4, 'morse_d0_g'),
                'r': (4, 5, 'morse_d0_r')
            }[color]
            pos1 = real_ppos[0][pos1_index]
            pos2 = real_ppos[1][pos2_index]
            morse_epsilon = args['morse_d0'] * args[d0_key]
            total_energy += calculate_morse_energy(pos1, pos2, morse_epsilon)

        return total_energy

 
        """
        pos1_b = real_ppos[0][3]
        pos2_b = real_ppos[1][3]
        morse_epsilon_b = args['morse_d0'] * args['morse_d0_b']
        morse_energy_fn_b = energy.morse_pair(displacement_fn,
                                           sigma=1.0,
                                           epsilon=morse_epsilon_b,
                                           alpha=morse_alpha,
                                           r_onset=morse_r_onset,
                                           r_cutoff=morse_r_cutoff)
        pos_b = jnp.array(pos1_b)
        pos_b.extend(pos2_b)
        morse_energy_b = morse_energy_fn_b(pos_b)
        tot_energy += morse_energy_b

     
        pos1_g = real_ppos[0][5]
        pos2_g = real_ppos[1][4]
        morse_epsilon_g = args['morse_d0'] * args['morse_d0_g']
        morse_energy_fn_g = energy.morse_pair(displacement_fn,
                                           sigma=1.0,
                                           epsilon=morse_epsilon_g,
                                           alpha=morse_alpha,
                                           r_onset=morse_r_onset,
                                           r_cutoff=morse_r_cutoff)
        pos_g = jnp.array(pos1_g)
        pos_g.extend(pos2_g)
        morse_energy_g = morse_energy_fn_g(pos_g)
        tot_energy += morse_energy_g

      
        pos1_r = real_ppos[0][4]
        pos2_r = real_ppos[1][5]
        morse_epsilon_r = args['morse_d0'] * args['morse_d0_r']
        morse_energy_fn_r = energy.morse_pair(displacement_fn,
                                           sigma=1.0,
                                           epsilon=morse_epsilon_r,
                                           alpha=morse_alpha,
                                           r_onset=morse_r_onset,
                                           r_cutoff=morse_r_cutoff)
        pos_r = jnp.array(pos1_r)
        pos_r.extend(pos2_r)
        morse_energy_r = morse_energy_fn_r(pos_r)
        tot_energy += morse_energy_r

        return tot_energy
        
        """
    return monomer_energy, cluster_energy

def add_variables(ma, mb):

    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb,Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans, angles))

def add_variables_all(mas, mbs):
  

    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))

    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(
        mas_temp, mbs_temp), mas.shape)


def setup_variable_transformation(energy_fn, q0, ppos):


    Nbb = q0.shape[0] // 6 
 

    E = energy_fn(q0, ppos)
    G = grad(energy_fn)(q0, ppos)
    H = hessian(energy_fn)(q0, ppos)

    evals, evecs = jnp.linalg.eigh(H)

    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(evals < zeromode_thresh, 1, 0))

    if Nbb == 1:
        zvib = 1.0
    else:
        zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))

    def ftilde(nu):
        return jnp.matmul(evecs.T[6:].T, nu[6:])

    def f_multimer(nu, addq0=True):
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
    mean = jnp.mean(Js)
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
