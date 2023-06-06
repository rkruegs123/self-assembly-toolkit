import pdb
from tqdm import tqdm
import random as rdm
import argparse
import numpy as np
import jax.numpy as jnp
from scipy.optimize import fsolve
from jax import random
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
from jax.ops import index, index_add, index_update

import potentials
from jax_transformations3d import jax_transformations3d as jts

from jax.config import config
config.update("jax_enable_x64", True)

# euler_scheme: string of 4 characters (e.g. 'sxyz') that define euler angles
euler_scheme = "sxyz"


def ConvertToMatrix(mi):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T, R)


def CalculateMomentOfInertiaDiag(e, positions):
    I = 0.
    for p in positions:
        d = np.dot(p,p) - (np.dot(p, e)**2) / np.dot(e, e)
        I += d
    return I

def CalculateMomentOfInertiaOffDiag(e1, e2, positions):
    I = 0.
    for p in positions:
        d = np.dot(p, e1) * np.dot(p, e2)
        I -= d
    return I

def CalculatePrincipalMomentsOfInertia(positions):
    Ixx = CalculateMomentOfInertiaDiag(np.array([1,0,0]), positions)
    Iyy = CalculateMomentOfInertiaDiag(np.array([0,1,0]), positions)
    Izz = CalculateMomentOfInertiaDiag(np.array([0,0,1]), positions)

    Ixy = CalculateMomentOfInertiaOffDiag(np.array([1,0,0]), np.array([0,1,0]), positions)
    Iyz = CalculateMomentOfInertiaOffDiag(np.array([0,1,0]), np.array([0,0,1]), positions)
    Izx = CalculateMomentOfInertiaOffDiag(np.array([0,0,1]), np.array([1,0,0]), positions)

    I = np.array([[Ixx,Ixy,Izx], [Ixy,Iyy,Iyz], [Izx,Iyz,Izz]])

    evals, evecs = np.linalg.eigh(I)

    return evals, evecs



class ParticleType:
    def __init__(self, typeString='AA', radius=1.0):
        self.typeString = typeString
        self.radius = radius

    def __str__(self):
        return f"PType: {self.typeString} {self.radius}"
    def __repr__(self):
        return f"PType: {self.typeString} {self.radius}"

class BuildingBlockType:
    def __init__(self):
        self.n = 0
        self.positions = jnp.array([])
        self.typeids = jnp.array([])
        self.mass = 1.0
        self.moment_inertia = jnp.array([])


class PotParam:
    def __init__(self, rmin=0, rmax=0, coeff=dict()):
        self.rmin = rmin
        self.rmax = rmax
        self.coeff = coeff

    def __str__(self):
        return f"PotParam {self.rmin} {self.rmax} {self.coeff}"
        # return ('PotParam: %f %f '%(self.rmin,self.rmax)) + str(self.coeff)
    def __repr__(self):
        return f"PotParam {self.rmin} {self.rmax} {self.coeff}"
        # return ('PotParam: %f %f '%(self.rmin,self.rmax)) + str(self.coeff)

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



class SystemDefinition:
    def __init__(self):
        self.buildingBlockTypeList = []
        self.particleTypes = []
        self.interactions = Interactions()
        self.L = 0.
        self.kBT = 0.
        self.seed = 12345
        self.basename = "test"


class CParams:
    # these are all class variables, they do not change in
    # different instances of the class CParams

    concentration = 0.000001
    ls = 150.

    Ttot = 1e8      # actually total number of steps. Ttot_real=Ttot
    Trec_data = 1000000
    Trec_traj = 1000000
    dT = 0.0001

    rep_A = 5.0
    rep_alpha = 2.5

    morse_D0 = 1.0

    morse_a = 5.0

    morse_r0 = 0.

    kT_brown = 1.0
    seed_brown = rdm.randint(1,1001)

    fnbase = "test"


def parserFunc():
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-i', '--input_name', type=str, default='test', help='input file')
    parser.add_argument('-D', '--morse_D0', type=float, default=7., help='morse min')
    parser.add_argument('-a', '--morse_a', type=float, default=5., help='morse strength')
    parser.add_argument('-c', '--conc', type=float, default=0.001, help='concentration')
    parser.add_argument('-s', '--string', type=str, default='test', help='output file name')
    parser.add_argument('-t', '--testing', action="store_true", default=False, help='put the simulation in testing mode')
    parser.add_argument('-f', '--save_file', action="store_true", default=False, help='save snaps of the clusters found')

    return parser



def InitBuildingBlock_sphere():

    inputf = 'ipack.3.60.txt'

    positions = np.loadtxt(inputf).reshape((60, 3))
    types = [f"S{i}" for i in range(len(positions))]
    mass = [1 for i in range(len(positions))]
    evals_I = [np.array([1, 1, 1]) for i in range(len(positions))]
    radii = [0.25 for i in range(len(positions))]

    neighbor_types = {}
    neighbor_positions = {}
    thresh = 0.5

    for (pos, typ) in zip(positions, types):
        neighbor_list_typ = []
        neighbor_list_pos = []
        for pos_n, typ_n in zip(positions, types):
            dist = np.linalg.norm(pos - pos_n)
            if dist > 0 and dist < thresh:
                neighbor_list_typ.append(typ_n)
                neighbor_list_pos.append(pos_n)
        neighbor_types[typ] = neighbor_list_typ
        neighbor_positions[typ] = neighbor_list_pos

    return (positions,types,mass,evals_I,radii,neighbor_types,neighbor_positions)



def InitializeSystem_sphere(params):

    Nbbt = 60

    (p,t,m,I,r,n_typ,n_pos) = InitBuildingBlock_sphere()

    PartTypes = [ParticleType() for i in range(len(t))]
    for i in range(len(t)):
        PartTypes[i].typeString = t[i]
        PartTypes[i].radius = r[i]

    # Create and populate the building block type
    BBlockTypeList = []

    count = 0

    for i in range(Nbbt):
        BBlockType = BuildingBlockType() # for i in range(n_bb)
        BBlockType.ID = i
        BBlockType.n = len(p[i])
        BBlockType.positions = np.array(p[i])
        BBlockType.typeids = i
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
    Inter.InitializeMatrix(len(t))

    morse_rcut = 8. / params.morse_a + params.morse_r0

    dist_ave = 0.4638568818178284 # found to be the average distance of the first nearest neighbors
    thresh = 0.75 # found to be the threshold before second nearest neighbor

    # Populate interaction matrix
    for pos1, typ1 in zip(p, t):
        i = int(typ1[1:])
        for pos2, typ2 in zip(p, t):
            j = int(typ2[1:])
            dist = np.linalg.norm(np.array(pos1 - pos2))
            if dist > 0:
                Inter.matrix[i][j][1] = PotParam().SetMorseX(
                    0, thresh, params.morse_D0, params.morse_a, dist, 0)

    # Create and populate the system definition object
    SystDef = SystemDefinition()
    SystDef.buildingBlockTypeList = BBlockTypeList
    SystDef.particleTypes = PartTypes
    SystDef.interactions = Inter
    SystDef.concentration = params.concentration
    SystDef.Lxyz = 10.
    SystDef.kBT = params.kT_brown
    SystDef.seed = rdm.randint(1, 1001)
    SystDef.basename = params.fnbase
    SystDef.neigh_t = n_typ
    SystDef.neigh_p = n_pos

    return SystDef



def setup_system(SystDef):

    BBt = SystDef.buildingBlockTypeList

    positions = jnp.array([bb.positions for bb in BBt])

    t_dict = SystDef.neigh_t
    p_dict = SystDef.neigh_p

    # the particle positions are in
    ref_ppos_tot = []
    bb_tot = []
    ref_q0_tot = []
    last_struct_tot = []

    # change this to n < 60 to test
    while len(ref_ppos_tot) < 60:

        if len(ref_ppos_tot) == 0:
            [(one_random_id, one_random_pos)] = rdm.sample(list(enumerate(positions)), 1)

            ref_ppos = [jnp.array([0,0,0])]
            last_struct = [one_random_pos[:]]
            last_struct_bb = [one_random_id]
            last_struct_q0 = jnp.array([one_random_pos[0], one_random_pos[1], one_random_pos[2], 0, 0, 0], dtype=jnp.float64) + 1e-14

            ref_ppos_tot.append(ref_ppos[:])
            last_struct_tot.append(last_struct[:])
            bb_tot.append(last_struct_bb[:])
            ref_q0_tot.append(jnp.array(last_struct_q0))

        if len(ref_ppos_tot) == 1:
            for p_id, p in enumerate(positions):
                dist0 = jnp.linalg.norm(jnp.array(one_random_pos) - jnp.array(p))
                if dist0 < 0.5 and dist0 > 0:
                    ref_ppos.append(jnp.array([0, 0, 0]))
                    last_struct.append(p[:])
                    last_struct_bb.append(p_id)
                    last_struct_q0 = jnp.concatenate(
                        (jnp.array(last_struct_q0),
                         jnp.array([p[0], p[1], p[2], 0, 0, 0], dtype=jnp.float64) + 1e-14))

                    ref_ppos_tot.append(ref_ppos[:])
                    last_struct_tot.append(last_struct[:])
                    bb_tot.append(last_struct_bb[:])
                    ref_q0_tot.append(jnp.array(last_struct_q0))

                    break
        else:
            last_struct = last_struct_tot[-1][:]
            two_random_pos = rdm.sample(last_struct, 2)
            dist0 = jnp.linalg.norm(two_random_pos[0] - two_random_pos[1])

            if dist0 > 0.5:
                continue
            else:
                last_pos = two_random_pos

            for p_id, p in enumerate(positions):
                dist1 = jnp.linalg.norm(p - last_pos[0])
                dist2 = jnp.linalg.norm(p - last_pos[1])

                array_dist = jnp.linalg.norm(jnp.array(last_struct) - p, axis=1)

                if jnp.all(array_dist) and dist1 < 0.5 and dist2 < 0.5:

                    ref_ppos.append(jnp.array([0, 0, 0]))
                    last_struct.append(p[:])
                    last_struct_bb.append(p_id)
                    last_struct_q0 = jnp.concatenate((
                        jnp.array(last_struct_q0),
                        jnp.array([p[0], p[1], p[2], 0, 0, 0], dtype=jnp.float64) + 1e-14))

                    ref_ppos_tot.append(ref_ppos[:])
                    last_struct_tot.append(last_struct[:])
                    bb_tot.append(last_struct_bb[:])
                    ref_q0_tot.append(jnp.array(last_struct_q0))

                    print("len struct list",len(ref_ppos_tot))

                    break


    def cluster_energy(q, ppos, BBlocks):
        """
        Calculate the energy of a cluster as a function of the 6N "q" variables
        """

        Inter = SystDef.interactions

        # q is later called as ref_q0 and it's a 6N array of
        # N*(x,y,z,alpha,beta,gamma) of each building block
        Nbb = len(BBlocks)
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

        if Nbb == 1:
            return tot_energy
        else:
            for bb1 in range(len(BBlocks)):
                for bb2 in range(bb1+1, len(BBlocks)):

                    t1 = BBt[BBlocks[bb1]].typeids
                    t2 = BBt[BBlocks[bb2]].typeids

                    if Inter.matrix[t1][t2][1] != None: # and bb1!=bb2:
                        pos1 = real_ppos[bb1]
                        pos2 = real_ppos[bb2]

                        rmin  = Inter.matrix[t1][t2][1].rmin
                        rmax  = Inter.matrix[t1][t2][1].rmax
                        coeff = list(Inter.matrix[t1][t2][1].coeff.values())
                        r = jnp.linalg.norm(pos1-pos2)

                        ID1 = SystDef.particleTypes[t1]
                        ID2 = SystDef.particleTypes[t2]

                        etemp = Inter.potentials[1].E(
                            r, rmin, rmax, coeff[0], coeff[1], coeff[2], coeff[3])

                        tot_energy += etemp

        return tot_energy

    return cluster_energy, ref_q0_tot, ref_ppos_tot, bb_tot


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


def setup_variable_transformation(energy_fn, q0, ppos, BBlocks):
    """
    Args:
    energy_fn: function to calculate the energy:
    E = energy_fn(q, euler_scheme, ppos)
    euler_scheme: string of 4 characters (e.g. 'sxyz') that define euler angles
    q0: initial coordinates (positions and orientations) of the building blocks
    ppos: "patch positions", array of shape (N_bb, N_patches, dimension)

    Returns: function f that defines the coordinate transformation, as well as
    the number of zero modes (which should be 6) and Z_vib

    Note: we assume without checking that ppos is defined such that all the euler
    angels in q0 are initially 0.
    """

    Nbb = q0.shape[0] // 6 # Number of building blocks
    assert(Nbb*6 == q0.shape[0])
    assert(len(ppos) == Nbb)
    assert(ppos[0].shape[0] == 3)

    E = energy_fn(q0, ppos, BBlocks)
    G = grad(energy_fn)(q0, ppos, BBlocks)
    H = hessian(energy_fn)(q0, ppos, BBlocks)

    print("Energy", E)
    print("Gradient", G)

    evals, evecs = jnp.linalg.eigh(H)

    print("evals", evals)

    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(abs(evals) < zeromode_thresh, 1, 0))
    non_zero_modes_id = jnp.where(abs(evals) > zeromode_thresh)

    print("num_zero_modes", num_zero_modes)

    evals = evals[non_zero_modes_id]
    evecs = evecs[non_zero_modes_id]

    if Nbb == 1:
        zvib = 1.0
    else:
        zvib = jnp.product(jnp.sqrt(2. * jnp.pi / (jnp.abs(evals) + 1e-12)))

    def ftilde(nu):
        return jnp.matmul(evecs.T[6:].T, nu[6:])

    def f_multimer(nu, addq0=True):
        dq_tilde = ftilde(nu)
        if (addq0 == False):
            q_tilde = dq_tilde
        else:
            q_tilde = add_variables_all(q0, dq_tilde)

        nu_bar_repeat = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return add_variables_all(q_tilde, nu_bar_repeat)

    def f_monomer(nu, addq0=True):
        return nu

    if Nbb == 1:
        f = f_monomer
    else:
        f = f_multimer

    return f, num_zero_modes, zvib


def Calculate_Zc(key, energy_fn, ref_q0, ref_ppos, BBlocks, sigma, kBT, V):
    """
    Calculate Zc except without the lambdas
    """

    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, ref_q0, ref_ppos, BBlocks)

    # Js_mean, Js_error = GetJmean_method1(f, key, len(BBlocks))

    Nbb = len(BBlocks)

    Mat = []
    for i in range(Nbb):
        qi = i*6
        Mat.append(ConvertToMatrix(ref_q0[qi:qi+6]))

    # apply building block matrix to spheres positions
    real_ppos = []
    for i in range(Nbb):
        real_ppos.append(jts.matrix_apply(Mat[i], ref_ppos[i]))

    if len(BBlocks) == 1:
        evals_I = jnp.array([1, 1, 1])
    else:
        evals_I, evecs_I = CalculatePrincipalMomentsOfInertia(real_ppos)

    print("evals_I", evals_I)

    # Jtilde = 8.0*(jnp.pi**2)*Js_mean
    Jtilde = 8.0*(jnp.pi**2) * jnp.sqrt(jnp.prod(evals_I))

    print("Jtilde", Jtilde)
    print("Zvib", zvib, "\n")

    E0 = energy_fn(ref_q0, ref_ppos, BBlocks)
    boltzmann_weight = jnp.exp(-E0/kBT)

    lnZc = -E0/kBT + jnp.log(zvib) + jnp.log(V) + jnp.log(Jtilde/sigma)
    Zc = jnp.exp(lnZc)

    return lnZc


# Constrain concentrations to all be positive by solving in log-space
def cons_law(multimer_log_concs, conc_total):
    # multimer_log_concs is a length-m array giving the log-concentration of each m-mer
    # Returns N0 - (N_1 + 2 N_2 + 3 N_3 + ... + m N_m)
    return (conc_total - np.sum(np.arange(1, 1 + len(multimer_log_concs)) *
                                np.exp(multimer_log_concs)))

def eq_Z_ratios(multimer_log_concs, m, log_Z, V):
    # Returns log(N_1^m / Z_1^m) - log(N_m rhoH20^(m-1) / Z_m) for the m specified
    # Working in log space is easier
    # m is actually the number of building blocks - 1 (i.e. m=1 is dimers, m=2 is trimers, etc.)
    return((m+1) * (np.log(V) + multimer_log_concs[0] - log_Z[0]) \
           - (np.log(V) + multimer_log_concs[m] - log_Z[m]))

def canonical_eqs(multimer_log_concs, conc_total, log_Z, V):
    # Can normalize all Z's by V and then set V to 1. This just makes Z smaller and easier to handle
    log_Z_norm = np.array(log_Z) - np.log(V)
    VOne = 1
    return(np.array(
        [cons_law(multimer_log_concs, conc_total=conc_total),] +
        [eq_Z_ratios(multimer_log_concs, m=m, log_Z=log_Z_norm, V=VOne)
         for m in range(1, len(multimer_log_concs))]
    ))


def calculate_concentrations_and_yields(
        conc_total, V, log_Z, max_m=60, num_guesses_to_try=1000,
        print_results=True
):

    """

    Parameters
    ----------
    conc_total : float
        Total concentration of monomers.
    V : float
        Volume of system.
    log_Z : array
        Log of partition function fo each multimer.
    max_m : int, optional
        Maximum multimer size. The default is 60.
    num_guesses_to_try : int, optional
        Maximum number of random initializations to try. The default is 1000.
    print_results : boolean, optional
        Whether or not to print concentrations and yields. The default is True.

    Returns
    -------
    multimer_concs_sol : array
        Concentrations of each multimer species in equilibrium
    yields : array
        Yields of each multimer species in equilibrium

    """

    for guess in range(num_guesses_to_try):
        multimer_concs_guess_unnormalized = np.random.random(max_m)
        multimer_log_concs_guess = np.log(
            multimer_concs_guess_unnormalized * conc_total / np.sum(
                np.arange(1, max_m + 1) * multimer_concs_guess_unnormalized))

        multimer_concs_sol = np.exp(fsolve(
                canonical_eqs, multimer_log_concs_guess, args=(
                        conc_total, log_Z, V)))

        # Check that we were able to satisfy the equations
        multimer_concs_check = canonical_eqs(  # should be zero if the solution we found is a real solution
                np.log(multimer_concs_sol),
                conc_total, log_Z, V)

        if not np.all(np.isclose(multimer_concs_check, 0)):
            if guess == num_guesses_to_try - 1:
                print('Was not able to satisfy all the equations simultaneously for guess # ' + str(guess))
        else:
            break

    yields = multimer_concs_sol / np.sum(multimer_concs_sol)

    if print_results:
        print('Concentrations: ')
        print(multimer_concs_sol)
        print('Yields: ')
        print(yields)

    return(multimer_concs_sol, yields)


def Full_Calculation(systdef, seed=0):

    # setup_system returns the energy function and three lists corresponding to:
    # - ref_q0: 6N-dimensional COM coordinates of the BBs in the structure
    # - ref_ppos: the positions of the spheres of a building block wrt the bb COM
    # - bb: the ID of the building blocks in the cluster

    key = random.PRNGKey(seed)

    conc = systdef.concentration
    N = 3
    v = N / conc

    cluster_energy, ref_q0_list, ref_ppos_list, bb_list = setup_system(systdef)

    Nc = len(bb_list)

    split = random.split(key,Nc)

    Zc_list = []

    for c in tqdm(range(Nc)):
        print("\nstruct", len(bb_list[c]))
        if len(bb_list[c]) == 2 or len(bb_list[c]) == 4:
            s = 2
        if len(bb_list[c]) == 3:
            s = 3
        elif len(bb_list[c]) == 60:
            s = 60
        else:
            s = 1
        zc = Calculate_Zc(
            split[c],
            cluster_energy,
            ref_q0_list[c],
            ref_ppos_list[c],
            bb_list[c],
            sigma=s,
            kBT=1.0,
            V=v
        )
        Zc_list.append(zc)

    conc_list = [conc for c in range(len(bb_list))]
    Nc_list = [len(bbs) for bbs in bb_list]

    eq_multimer_concs, eq_yields = calculate_concentrations_and_yields(
        conc,
        v,
        Zc_list,
        max_m=60,
        num_guesses_to_try=1000,
        print_results=True)

    return Zc_list, eq_multimer_concs, eq_yields


if __name__ == "__main__":
    params = CParams()

    parser = parserFunc()

    args = parser.parse_args()

    params.inputfile = '%s' % (args.input_name)

    params.concentration = args.conc
    params.morse_D0 = args.morse_D0

    # file where the trajectory is saved
    params.fnbase = f"temp_results/{args.string}_D0{args.morse_D0}_c{args.conc}"

    systdef = InitializeSystem_sphere(params)

    Zs, Cs, Ys = Full_Calculation(systdef)

    pdb.set_trace()

    print("done")
