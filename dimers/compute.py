import pdb
from tqdm import tqdm
import argparse
import numpy as np
import scipy as osp
import jax.numpy as jnp
import csv
from jax import random
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
from random import randint
from jax.ops import index, index_add, index_update


import potentials
from jax_transformations3d import jax_transformations3d as jts

from jax.config import config
config.update("jax_enable_x64", True)


# euler_scheme: string of 4 characters (e.g. 'sxyz') that define euler angles
euler_scheme = "sxyz"

class SystemDefinition:
    def __init__(self):
        self.buildingBlockTypeList = []
        self.particleTypes = []
        self.interactions = Interactions()
        # self.buildingBlockNumbers = []
        self.L = 0.
        self.kBT = 0.
        self.seed = 12345
        self.basename = "test"



class PotParam:
    def __init__(self, rmin=0, rmax=0, coeff=dict()):
        self.rmin = rmin
        self.rmax = rmax
        self.coeff = coeff

    def __str__(self):
        return f"PotParam: {self.rmin} {self.rmax} {self.coeff}"
        # return ('PotParam: %f %f ' % (self.rmin, self.rmax)) + str(self.coeff)
    def __repr__(self):
        return f"PotParam: {self.rmin} {self.rmax} {self.coeff}"
        # return ('PotParam: %f %f ' % (self.rmin, self.rmax)) + str(self.coeff)

    def SetRepulsive(self, rmin, rmax, A, alpha):
        self.rmin = rmin
        self.rmax = rmax
        self.coeff = dict(A=A, alpha=alpha)
        return self

    def SetMorseX(self, rmin, rmax, D0, alpha, r0, ron):
        self.rmin = rmin
        self.rmax = rmax
        self.coeff = dict(D0=D0, alpha=alpha, r0=r0, ron=ron)
        return self


class Interactions:
    def __init__(self, potentials=[], n_t=0):
        self.potentials = potentials
        self.InitializeMatrix(n_t)

    def InitializeMatrix(self, n_t):
        npotentials = len(self.potentials)
        self.matrix = [
            [
                [None for i in range(npotentials)] for j in range(n_t)
            ] for k in range(n_t)
        ]


class ParticleType:
    def __init__(self, typeString='AA', radius=1.0):
        self.typeString = typeString
        self.radius = radius

    def __str__(self):
        return f"PType: {self.typeString} {self.radius}"
        # return "PType: "+ self.typeString +" " + str(self.radius)
    def __repr__(self):
        return f"PType: {self.typeString} {self.radius}"
        # return "PType: "+ self.typeString +" " + str(self.radius)

class BuildingBlockType:
    def __init__(self):
        self.n = 0
        self.positions = jnp.array([])
        self.typeids = jnp.array([])
        self.mass = 1.0
        self.moment_inertia = jnp.array([])


def parserFunc():
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-i', '--input_name', type=str,  default='test', help='input file')
    parser.add_argument('-D', '--morse_D0',  type=float, default=1., help='morse overall')
    parser.add_argument('-DX', '--morse_D0_X', type=float, default=1., help='morse interface X')
    parser.add_argument('-DY', '--morse_D0_Y', type=float, default=1., help='morse interface Y')
    parser.add_argument('-DZ', '--morse_D0_Z', type=float, default=1., help='morse interface Z')
    parser.add_argument('-a', '--morse_a', type=float, default=5., help='morse strength')
    parser.add_argument('-c', '--conc', type=float, default=0.001, help='concentration')
    parser.add_argument('-s', '--string', type=str, default='test', help='output file name')
    parser.add_argument('-t', '--testing', action="store_true", default=False, help='put the simulation in testing mode')
    parser.add_argument('-f', '--save_file', action="store_true", default=False, help='save snaps of the clusters found')

    return parser


class CParams_dimer:
    # these are all class variables, they do not change in
    # different instances of the class CParams
    N = [9, 9]
    concentration = 0.001
    ls = 10.

    Ttot = 1e8      # actually total number of steps. Ttot_real=Ttot
    Trec_data = 1000000
    Trec_traj = 1000000
    dT = 0.0001

    rep_A = 500.0
    rep_alpha = 2.5

    morse_D0 = 10.
    morse_D0_r = 1. # 1.0
    morse_D0_g = 1. # 1.5
    morse_D0_b = 1. # 2.0

    morse_a = 5.0

    morse_r0 = 0.

    kT_brown = 1.0

    fnbase = "test"


def InitBuildingBlockDimers_tr(Type):

    a = 1 # distance of the center of the spheres from the BB COM
    b = .3 # distance of the center of the patches from the BB COM

    positions = [
        [0.,                  0.,                   a], # first sphere
        [0.,  a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)], # second sphere
        [0., -a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)], # third sphere
        [a,                   0.,                   b], # first patch
        [a,   b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)], # second patch
        [a,  -b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)]  # third patch
    ]

    if Type == 0:
        types = ["A", "A", "A", "B1", "R1", "G1"]
    elif Type == 1:
        types = ["A", "A", "A", "B2", "G2", "R2"]
    radii = [a, a, a, 0.2*a, 0.2*a, 0.2*a]
    mass = 1

    # used on simulations of 8-4-2020
    evals_I = np.array([3., 1.5, 1.5])

    COM = [np.mean(positions[i],axis=0) for i in range(3)]

    return (positions, types, mass, evals_I, radii, COM)



def InitializeSystem_dimers_tr(params):

    p = []
    t = []
    m = []
    I = []
    r = []

    for i in range(2):
        (p_temp, t_temp, m_temp, I_temp, r_temp, pCOM) = InitBuildingBlockDimers_tr(i)
        p.append(p_temp)
        t.append(t_temp)
        m.append(m_temp)
        I.append(I_temp)
        r.append(r_temp)

    # List of unique types, the corresponding indices, and the
    t_unique, i_unique, typeids = np.unique(t, return_index=True, return_inverse=True)

    # Empty list of radii
    rsum = []
    for i in range(len(r)):
        rsum += r[i]

    r_unique = [rsum[i] for i in i_unique]

    # Number of unique types
    n_t = len(t_unique)

    # Create and populate the particle types with type name and radius
    PartTypes = [ParticleType() for i in range(n_t)]
    for i in range(n_t):
        # Create and populate the particle types with type name and radius
        PartTypes[i].typeString = t_unique[i]
        PartTypes[i].radius = r_unique[i]

    # Create and populate the building block type: in this case only one
    BBlockTypeList = []

    count = 0

    for i in range(2):
        BBlockType = BuildingBlockType() # for i in range(n_bb)
        BBlockType.n = len(p[i])
        BBlockType.positions = np.array(p[i])
        BBlockType.typeids = typeids[count:count+BBlockType.n]
        BBlockType.mass = m[i]
        BBlockType.moment_inertia = I[i]

        BBlockTypeList.append(BBlockType)

        count += BBlockType.n

    # List of potentials
    Pots = [
        potentials.RepulsivePotential(),
        potentials.MorseXPotential(),
        potentials.MorseXRepulsivePotential()
    ]

    # Create the interaction object
    Inter = Interactions()
    Inter.potentials = Pots
    Inter.InitializeMatrix(n_t)

    morse_rcut = 8. / params.morse_a + params.morse_r0


    # Populate interaction matrix
    for i in range(n_t):
        for j in range(n_t):
            if (t_unique[i] == t_unique[j] and t_unique[i] == 'A'):
                # Repulsion between real spheres
                Inter.matrix[i][j][0] = PotParam().SetRepulsive(
                    0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)

            elif (t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'R'):
                # Attraction between red patches
                Inter.matrix[i][j][1] = PotParam().SetMorseX(
                    0, morse_rcut, params.morse_D0*params.morse_D0_r,
                    params.morse_a, params.morse_r0, morse_rcut/2.)

            elif (t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'G'):
                # Attraction between green patches
                Inter.matrix[i][j][1] = PotParam().SetMorseX(
                    0, morse_rcut, params.morse_D0*params.morse_D0_g,
                    params.morse_a, params.morse_r0, morse_rcut/2.)

            elif (t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'B'):
                # Attraction between blue patches
                Inter.matrix[i][j][1] = PotParam().SetMorseX(
                    0, morse_rcut, params.morse_D0*params.morse_D0_b,
                    params.morse_a, params.morse_r0, morse_rcut/2.)

            elif (t_unique[i] == t_unique[j] and t_unique[i] != 'A'):
                # Repulsion between patches of the same type (e.g. two G1's)
                Inter.matrix[i][j][2] = PotParam().SetMorseX(
                    0, morse_rcut, params.morse_D0,
                    params.morse_a, params.morse_r0, morse_rcut/2.)

    # Create and populate the system definition object
    SystDef = SystemDefinition()
    SystDef.buildingBlockTypeList = BBlockTypeList
    SystDef.particleTypes = PartTypes
    SystDef.interactions = Inter
    SystDef.concentration = params.concentration
    SystDef.Lxyz = 10.
    SystDef.kBT = params.kT_brown
    SystDef.seed = randint(1, 1001)
    SystDef.basename = params.fnbase

    return SystDef


def ConvertToMatrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3],mi[4],mi[5], axes=euler_scheme)
    return jnp.matmul(T,R)

def setup_system_dimers_tr(SystDef):

    BBt = SystDef.buildingBlockTypeList

    ### just for now ###
    ### these are the positions of the spheres within the building block
    ref_ppos1 = BBt[0].positions
    ref_ppos2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0,0,0],dtype=jnp.float64),
                                                       jnp.array([1,0,0],dtype=jnp.float64)),
                                 ref_ppos1
                             )

    ref_ppos2 = jts.matrix_apply(jts.reflection_matrix(jnp.array([0,0,0],dtype=jnp.float64),
                                                       jnp.array([0,1,0],dtype=jnp.float64)),
                                 ref_ppos2
    )

    ref_ppos = jnp.array([ref_ppos1, ref_ppos2])

    def monomer_energy(q, ppos):
        assert(q.shape[0] == 6)
        return jnp.float64(0)

    def cluster_energy(q, ppos):
        """
        Calculate the energy of a dimer as a function of the 12 "q" variables
        """

        Inter = SystDef.interactions

        # q is later called as ref_q0 and it's a 6N array of
        # N*(x,y,z,alpha,beta,gamma) of each building block
        Nbb = len(BBt)
        assert(Nbb == q.shape[0] // 6)

        # convert the building block coordinates to a tranformation
        # matrix
        Mat = []
        for i in range(Nbb):
            qi = i*6
            Mat.append(ConvertToMatrix(q[qi:qi+6]))

        # apply building block matrix to spheres positions
        real_ppos = []
        for i in range(Nbb):
            real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

        tot_energy = jnp.float64(0)

        ## the first set of two for loops must work with state building blocks
        for bb1 in range(2):
            for bb2 in range(bb1, 2):
                for p1, t1 in enumerate(BBt[bb1].typeids):
                    for p2, t2 in enumerate(BBt[bb2].typeids):
                        for pi in range(3):
                            if Inter.matrix[t1][t2][pi] != None and bb1 != bb2:
                                pos1 = real_ppos[bb1][p1]
                                pos2 = real_ppos[bb2][p2]
                                rmin  = Inter.matrix[t1][t2][pi].rmin
                                rmax  = Inter.matrix[t1][t2][pi].rmax
                                coeff = list(Inter.matrix[t1][t2][pi].coeff.values())
                                r = jnp.linalg.norm(pos1-pos2)
                                if len(coeff) == 2:
                                    tot_energy += Inter.potentials[pi].E(
                                        r, rmin, rmax, coeff[0], coeff[1])
                                if len(coeff) == 4:
                                    tot_energy += Inter.potentials[pi].E(
                                        r, rmin, rmax, coeff[0], coeff[1], coeff[2], coeff[3])

        return tot_energy

    return monomer_energy, cluster_energy, ref_ppos

def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = ConvertToMatrix(ma)
    Mb = ConvertToMatrix(mb)
    Mab = jnp.matmul(Mb,Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans,angles))

def add_variables_all(mas, mbs):
    """
    Given two vectors of length (6*n,), 'add' them per building block according
    to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0]//6,6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0]//6,6))


    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(
        mas_temp, mbs_temp), mas.shape)


def setup_variable_transformation_old(energy_fn, q0, ppos):
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
    num_zero_modes = jnp.sum(jnp.where(evals<zeromode_thresh, 1, 0))

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


def GetJmean_method1_old(f, key, nrandom=100000):
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


def Calculate_Zc_old(key, energy_fn, ref_q0, ref_ppos, sigma, kBT, V):
    """
    Calculate Zc except without the lambdas
    """

    f, num_zero_modes, zvib = setup_variable_transformation_old(energy_fn, ref_q0, ref_ppos)

    Js_mean, Js_error = GetJmean_method1_old(f, key)
    Jtilde = 8.0*(jnp.pi**2) * Js_mean

    E0 = energy_fn(ref_q0, ref_ppos)
    boltzmann_weight = jnp.exp(-E0/kBT)

    print("E0", len(ref_q0),E0)
    print("zvib", len(ref_q0),zvib)
    print("Jtilde", len(ref_q0),Jtilde)

    return boltzmann_weight * V * (Jtilde/sigma) * zvib
    # return boltzmann_weight * V


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

def Full_Calculation_can(params, systdef, seed=0):
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

    monomer_energy, dimer_energy, ref_ppos = setup_system_dimers_tr(systdef)

    [Nblue, Nred] = params.N

    conc = params.concentration
    Ntot = jnp.sum(jnp.array(params.N))
    V = Ntot / conc

    separation = 2.

    ref_q0 = jnp.array([-separation/2.0,1e-16,0,0,0,0,separation/2.0,0,0,0,0,0], dtype=jnp.float64)

    split1, split2 = random.split(key)
    Zc_dimer = Calculate_Zc_old(
        split1, dimer_energy, ref_q0, ref_ppos,
        sigma=1, kBT=1.0, V=V)
    Zc_monomer = Calculate_Zc_old(
        split2, monomer_energy,
        ref_q0[:6], jnp.array([ref_ppos[0]]),
        sigma=1, kBT=1.0, V=V)


    # these will need to change for trimers
    pc_list = Calculate_pc_list(Nblue, Nred, Zc_monomer, Zc_dimer, exact=True)
    Y_dimer = Calculate_yield_can(Nblue, Nred, pc_list)

    return Y_dimer, pc_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time


    params = CParams_dimer()
    parser = parserFunc()
    args = parser.parse_args()

    params.inputfile = '%s' % (args.input_name)

    params.concentration = args.conc

    # all_eb = np.linspace(4, 12, 6)
    all_eb = np.linspace(4, 12, 20)
    all_yields = list()

    start = time.time()
    for d0 in tqdm(all_eb):
        # params.morse_D0 = args.morse_D0
        params.morse_D0 = d0

        # file where the trajectory is saved
        params.fnbase = 'temp_results/%s_D0%s_c%s' % (args.string,str(args.morse_D0), str(args.conc))

        systdef = InitializeSystem_dimers_tr(params)

        Ys, pc = Full_Calculation_can(params, systdef)
        all_yields.append(Ys)

        print(Ys)
        print(pc)
    end = time.time()

    print(f"Total execution: {end - start} seconds")

    plt.plot(all_eb, all_yields)
    plt.show()

    pdb.set_trace()



    print("done")
