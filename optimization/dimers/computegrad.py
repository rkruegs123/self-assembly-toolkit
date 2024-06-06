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

#from jax.config import config
#config.update("jax_enable_x64", True)



@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
  masked = jnp.where(mask, operand, 0)
  return jnp.where(mask, fn(masked), placeholder)

def distance(dR):
  dr = jnp.sum(dR ** 2, axis=-1)
  return safe_mask(dr > 0, jnp.sqrt, dr)


# dist_fn = jnp.linalg.norm
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
        # assert(q.shape[0] == 6)
        return jnp.float64(0)


    sphere_radius = 1.0
    patch_radius = 0.2 * sphere_radius
    # types: ['A', 'B1', 'B2', 'G1', 'G2', 'R1', 'R2']
    # BBt[0].typeids: array([0, 0, 0, 1, 5, 3])
    # BBt[1].typeids: array([0, 0, 0, 2, 4, 6])

    morse_rcut = 8. / args['morse_a'] + args['morse_r0']
    def cluster_energy(q, ppos):
        # convert the building block coordinates to a tranformation matrix

        # get_trans_matrix_elt = lambda i: convert_to_matrix(q[i*6:i*6+6])
        # Mat = vmap(get_trans_matrix_elt)(jnp.arange(Nbb))

        Mat = []
        for i in range(Nbb):
            qi = i*6
            Mat.append(convert_to_matrix(q[qi:qi+6]))

        # apply building block matrix to spheres positions

        # get_transf_pos = lambda i: jts.matrix_apply(Mat[i], ppos[i])
        # real_ppos = vmap(get_transf_pos)(jnp.arange(Nbb))
        
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

    E0 = energy_fn(all_q0, all_ppos)
    boltzmann_weight = jnp.exp(-E0/kBT)

    # print("E0", len(all_q0), E0)
    # print("zvib", len(all_q0), zvib)
    # print("Jtilde", len(all_q0), Jtilde)

    return boltzmann_weight * V * (Jtilde/sigma) * zvib
    # return boltzmann_weight * V * (Jtilde/sigma)



N_mon_real = 9
def Calculate_pc_list(N_mon, Zc_monomer, Zc_dimer, exact=False):
    # nd_fact = jax_factorial(N_mon_real)

    def Mc(Nd):
        return comb_table[N_mon_real, Nd] * comb_table[N_mon_real, Nd] * factorial_table[Nd]

    def Pc(Nd):
        return Mc(Nd) * (Zc_dimer**Nd) * (Zc_monomer**(N_mon_real-Nd)) * (Zc_monomer**(N_mon_real-Nd))

    pc_list = vmap(Pc)(jnp.arange(N_mon_real+1))
    return pc_list / jnp.sum(pc_list)



"""
def Calculate_pc_list(N_mon, Zc_monomer, Zc_dimer, exact=False):
    def Mc(Nd):
        return osp.special.comb(Nb, Nd, exact=exact) \
            * osp.special.comb(Nr, Nd, exact=exact) \
            * osp.special.factorial(Nd, exact=exact)

    def Pc(Nd):
        return Mc(Nd) * (Zc_dimer**Nd) * (Zc_monomer**(Nb-Nd)) * (Zc_monomer**(Nr-Nd))

    pc_list = jnp.array([Pc(Nd) for Nd in range(0, N_mon+1)])
    pc_list = pc_list / jnp.sum(pc_list)

    return pc_list
"""


# NOTE: the following code relaxes the assumption that Nb == Nr, but is not immediately jit-able. We could fix this if we want, but we don't care for now
"""
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
"""


def Calculate_yield_can(Nb_dummy, Nr_dummy, pc_list):
    Nb = 9
    Nr = 9

    Y_list = vmap(lambda Nd: Nd / (Nb+Nr-Nd))(jnp.arange(N_mon_real+1))
    return jnp.dot(Y_list, pc_list)

"""
def Calculate_yield_can(Nb, Nr, pc_list):
    Y_list = jnp.array([Nd / (Nb+Nr-Nd) for Nd in range(len(pc_list))])
    return jnp.dot(Y_list, pc_list)
"""

def run(args, noise_terms, seed=0):
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

    # print( monomer_energy, dimer_energy)

    Nblue, Nred = args['num_monomer'], args['num_monomer']

    conc = args['conc']
    Ntot = jnp.sum(jnp.array(args['num_monomer']))
    V = Ntot / conc
    ref_q0 = setup_ref_q0(noise_terms)
    split1, split2 = random.split(key)
    Zc_dimer = calculate_zc(
        split1, dimer_energy, ref_q0, ref_ppos,
        sigma=1, kBT=1.0, V=V)
    Zc_monomer = calculate_zc(
        split2, monomer_energy,
        ref_q0[:6], jnp.array([ref_ppos[0]]),
        sigma=1, kBT=1.0, V=V)


    # Note: this one works
    # return Zc_dimer, None

    # Note: maybe this will work
    # pc_list = Calculate_pc_list(args['num_monomer'], Zc_monomer, Zc_dimer, exact=True)
    # return jnp.mean(pc_list), None

    # Note: what about this one?
    pc_list = Calculate_pc_list(args['num_monomer'], Zc_monomer, Zc_dimer, exact=True)
    Y_dimer = Calculate_yield_can(Nblue, Nred, pc_list)
    return Y_dimer
    

    # Note: this one doesn't work
    # pc_list = Calculate_pc_list(args['num_monomer'], Zc_monomer, Zc_dimer, exact=True)
    # Y_dimer = Calculate_yield_can(Nblue, Nred, pc_list)
    # return Y_dimer, pc_list
    
    

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

if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())






    def yield_fn(d0, args, seed):
        args['morse_d0'] = d0
        #args['morse_d0_r'] = dr
        #args['morse_d0_g'] = dg
        #args['morse_d0_b'] = db
        
        vmap_run = vmap(run, in_axes=(None, 0, None))
        noise_terms = jnp.array([1e-15, 1e-15,1e-14, 1e-14])
        yi_values = vmap_run(args, noise_terms, seed)
        #nans_replaced = jnp.where(jnp.isnan(yi_values), 0.0, yi_values)
        #non_zero_values = nans_replaced[nans_replaced != 0]
        sampled_yi = jnp.nanmean(yi_values)
        return sampled_yi


    def loss_fn(params, args, target_yield, seed):
        d0 = params['d0']
        yi = yield_fn(d0, args, seed)
        return (yi-target_yield)**2, yi

    grad_yield = jit(jacrev(loss_fn, has_aux=True))


    # Test 1: Do a single forward calculation
    """
    first_eval_start = time.time()
    grads, curr_yield = grad_yield(params, args, target_yield, seed=9)
    first_eval_end = time.time()
    print(f"First eval took: {first_eval_end - first_eval_start} seconds")

    second_eval_start = time.time()
    grads, curr_yield = grad_yield(params, args, target_yield, seed=9)
    second_eval_end = time.time()
    print(f"Second eval took: {second_eval_end - second_eval_start} seconds")


    pdb.set_trace()
    """


    # Test 2: Do a range of forward calculations
    """
    d0s = onp.arange(3, 13, 0.1)
    all_yields = list()
    all_grads = list()
    for d0 in tqdm(d0s):
        params = {'d0': d0}
        grads, curr_yield = grad_yield(params, args, target_yield, seed=9)
        all_yields.append(curr_yield)
        all_grads.append(grads['d0'])

    pdb.set_trace()

    
    plt.plot(d0s, all_yields)
    plt.savefig("yields.png")
    plt.clf()

    plt.plot(d0s, all_grads)
    plt.savefig("grads.png")
    plt.clf()

    pdb.set_trace()
    """
    

    d0 = 10.0
    target_yield = 0.4


    num_iters = 250
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    params = {'d0': d0}
    opt_state = optimizer.init(params)

    yield_path = "yield.txt"
    grad_path = "grads.txt"
    d0_path = "d0s.txt"
    
    open(yield_path, 'w').close()
    open(grad_path, 'w').close()
    open(d0_path, 'w').close()


    for i in tqdm(range(num_iters)):
        print(f"Iteration {i}:")

        grads, curr_yield = grad_yield(params, args, target_yield, seed=9)
        # grads = grads['d0']
        # grads_numeric = grads.item()
        curr_d0_grad = grads['d0']
        print(f"\t- grad: {float(curr_d0_grad)}")
        print(f"\t- current yield: {curr_yield}")

        curr_d0 = params['d0']
        print(f"\t- current d0: {float(curr_d0)}")

        with open(yield_path, "a") as f:
            f.write(f"{curr_yield}\n")
        with open(grad_path, "a") as f:
            f.write(f"{curr_d0_grad}\n")
        with open(d0_path, "a") as f:
            f.write(f"{curr_d0}\n")

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    pdb.set_trace()

    print("done")
