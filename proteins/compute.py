import argparse
import numpy as np
import random as rdm
import pdb
import pandas as pd
import scipy as osp

import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev
from jax import ops
from jax.ops import index, index_add, index_update
from jax import random

from jax.config import config
config.update("jax_enable_x64", True)

from potentials import *
import transformations
from jax_transformations3d import jax_transformations3d as jts


class ParticleType:
    def __init__(self, typeString='AA', radius=1.0):
        self.typeString = typeString
        self.radius = radius

    def __str__(self):
        return "PType: " + self.typeString + " " + str(self.radius)
    def __repr__(self):
        return "PType: " + self.typeString + " " + str(self.radius)

class BuildingBlockType:
    def __init__(self):
        self.n = 0
        self.positions = np.array([])
        self.typeids = np.array([])
        self.mass = 1.0
        self.moment_inertia = np.array([])

class Interactions:
    def __init__(self,potentials=[], n_t=0):
        self.potentials = potentials
        self.InitializeMatrix(n_t)

    def InitializeMatrix(self, n_t):
        npotentials = len(self.potentials)
        self.matrix = [[[None for i in range(npotentials)] for j in range(n_t)] for k in range(n_t)]

    def CheckSymmetric(self):
        assert(True)

class PotParam:
    def __init__(self, rmin=0, rmax=0, coeff=dict()):
        self.rmin = rmin
        self.rmax = rmax
        self.coeff = coeff

    def __str__(self):
        return ('PotParam: %f %f '%(self.rmin, self.rmax)) + str(self.coeff)
    def __repr__(self):
        return ('PotParam: %f %f '%(self.rmin, self.rmax)) + str(self.coeff)

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

    def GetTypeStrings(self):
        return np.array([ pt.typeString for pt in self.particleTypes ])

    # for a given building block, get the list of strings for the type of each particle
    def GetParticleTypeStrings(self, bbi):
        return np.array([self.particleTypes[tid].typeString for tid in self.buildingBlockTypeList[bbi].typeids])


    # for each particle in the building block
    # rotate by q and then translate by t
    def TransAndRotBB(self, t, q, bbT):
        Rrot = transformations.quaternion_matrix(q)
        Rtrans = transformations.translation_matrix(t)
        T = np.matmul(Rtrans,Rrot)
        return np.array([np.matmul(T, np.concatenate((p, [1])))[:3] for p in self.buildingBlockTypeList[bbT].positions])


    # translate by t and rotate by q particle at position p
    # p is the position of the particle relative to the COM of the building block it belongs to
    def TransAndRotPart(self, t, q, p):
        Rrot = transformations.quaternion_matrix(q)
        Rtrans = transformations.translation_matrix(t)
        T = np.matmul(Rtrans, Rrot)
        return np.array(np.matmul(T, np.concatenate((p, [1])))[:3])


# Parameters
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
    # seed_brown = rdm.randint(1,1001)

    fnbase = "test"


# Parameters
class CParams:
    # these are all class variables, they do not change in
    # different instances of the class CParams

    # when running YIA
    BBtypes = ['A', 'B', 'C']

    # which BBs are in contact with one another
    # e.g. if topology is A-B-C: topology = [[0, 1], [1, 2]]
    # e.g. if topology is A-B-C-D-A: topology = [[0, 1], [1, 2], [2, 3], [3, 1]]
    topology =  [[0, 2], [1, 2], [0, 2]]

    # number of interactive aminoacids per interface
    aa_interf = [20, 20, 20]
    # aa_interf = [10, 10, 10]

    N = [8, 8, 8]


    """
    # when running PFL
    BBtypes = ['A', 'B']
    topology =  [[0, 1]]
    aa_interf = [4]
    N = [8, 8]
    """

    sigma = [1, 1, 1, 1, 1, 1, 3]

    concentration = 0.000001
    ls = 150.

    Ttot = 1e8 # actually total number of steps. Ttot_real=Ttot
    Trec_data = 1000000
    Trec_traj = 1000000
    dT = 0.0001

    rep_A = 5.0
    rep_alpha = 2.5

    morse_D0 = 1.0

    morse_D0_X = 1.0 # interface MO or AC
    morse_D0_Y = 1.0 # interface NO or BC
    morse_D0_Z = 1.0 # interface MN or AB

    morse_a = 2.0

    morse_r0 = 0.

    kT_brown = 1.0
    seed_brown = rdm.randint(1, 1001)

    fnbase = "test"


def parserFunc():
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-i', '--input_name', type=str, default='test', help='input file')
    parser.add_argument('-D', '--morse_D0', type=float, default=1., help='morse overall')
    parser.add_argument('-DX', '--morse_D0_X', type=float, default=1., help='morse interface X')
    parser.add_argument('-DY', '--morse_D0_Y', type=float, default=1., help='morse interface Y')
    parser.add_argument('-DZ', '--morse_D0_Z', type=float, default=1., help='morse interface Z')
    parser.add_argument('-a', '--morse_a', type=float, default=5., help='morse strength')
    parser.add_argument('-c', '--conc', type=float, default=0.001, help='concentration')
    parser.add_argument('-s', '--string', type=str, default='test', help='output file name')
    parser.add_argument('-t', '--testing', action="store_true", default=False, help='put the simulation in testing mode')
    parser.add_argument('-f', '--save_file', action="store_true", default=False, help='save snaps of the clusters found')

    return parser




def CalculateMomentOfInertiaDiag(e, positions):
    # Isphere = (2./5.)*(0.5**2)  # should we include this?????
    I = 0.
    for p in positions:
        # d = math.sqrt(np.dot(p,p) - (np.dot(p,e)**2)/np.dot(e,e))
        d = np.dot(p, p) - (np.dot(p, e)**2)/np.dot(e, e)
        I += d
    return I

def CalculateMomentOfInertiaOffDiag(e1, e2, positions):
    # Isphere = (2./5.)*(0.5**2)  # should we include this?????
    I = 0.
    for p in positions:
        d = np.dot(p, e1) * np.dot(p, e2)
        I -= d
    return I

def CalculatePrincipalMomentsOfInertia(positions):
    Ixx = CalculateMomentOfInertiaDiag(np.array([1, 0, 0]), positions)
    Iyy = CalculateMomentOfInertiaDiag(np.array([0, 1, 0]), positions)
    Izz = CalculateMomentOfInertiaDiag(np.array([0, 0, 1]), positions)

    Ixy = CalculateMomentOfInertiaOffDiag(np.array([1, 0, 0]), np.array([0, 1, 0]), positions)
    Iyz = CalculateMomentOfInertiaOffDiag(np.array([0, 1, 0]), np.array([0, 0, 1]), positions)
    Izx = CalculateMomentOfInertiaOffDiag(np.array([0, 0, 1]), np.array([1, 0, 0]), positions)

    I = np.array([[Ixx, Ixy, Izx], [Ixy, Iyy, Iyz], [Izx, Iyz, Izz]])

    evals, evecs = np.linalg.eigh(I)

    return evals, evecs



def InitBuildingBlock_OPENSEQ_YIA(params):

    # building block types specified in params
    bbt = params.BBtypes
    Nbbt = len(bbt)

    # prepare array for building block amino acids positions
    pos = []

    for X in bbt:
        input_f = params.inputfile + X + '.pos'
        temp_pos = np.loadtxt(input_f, usecols=[1, 2, 3])
        pos.append(temp_pos)

    pos = np.array(pos, dtype=object)

    pos_COM = [np.mean(pos[i], axis=0) for i in range(Nbbt)]

    posA = pos[0]-pos_COM[0]
    posB = pos[1]-pos_COM[1]
    posC = pos[2]-pos_COM[2]

    # neutral particle types N
    typA = ['A' for i in range(len(pos[0]))]
    typB = ['B' for i in range(len(pos[1]))]
    typC = ['C' for i in range(len(pos[2]))]


    # FIRST INTERFACE AC [0,2], 20 aa interface MO X

    data_AC = pd.read_csv('OPENSEQ/YIAM-YIAO', sep='\t')

    data_AC = data_AC.dropna()

    i_id = np.array([int(iid[:-2]) for iid in data_AC['i_id'].values])
    j_id = np.array([int(jid[:-2]) for jid in data_AC['j_id'].values])

    AC_scores = np.array(data_AC['p_sco'].values)
    AC_scores = AC_scores[:params.aa_interf[0]]

    # print("AC_scores",len(AC_scores))

    aa_A = pos[0][i_id[:params.aa_interf[0]]-1]
    aa_C = pos[2][j_id[:params.aa_interf[0]]-26]

    # aa_A = pos[0][i_id-1]
    # aa_C = pos[2][j_id-26]

    AC_patch_pos = (aa_A+aa_C)/2

    posA1 = AC_patch_pos - pos_COM[0]
    posC1 = AC_patch_pos - pos_COM[2]

    typA1 = ['Xa_%s'%i for i in range(len(posA1))]
    typC1 = ['Xc_%s'%i for i in range(len(posC1))]


    # SECOND INTERFACE BC [1,2], 20 aa interface NO Y

    data_BC = pd.read_csv('OPENSEQ/YIAN-YIAO', sep='\t')

    data_BC = data_BC.dropna()

    i_id = np.array([int(iid[:-2]) for iid in data_BC['i_id'].values])
    j_id = np.array([int(jid[:-2]) for jid in data_BC['j_id'].values])

    BC_scores = np.array(data_BC['p_sco'].values)
    BC_scores = BC_scores[:params.aa_interf[1]]

    # print("BC_scores",len(BC_scores))

    aa_B = pos[1][i_id[:params.aa_interf[1]]-1]
    aa_C = pos[2][j_id[:params.aa_interf[1]]-26]

    # aa_B = pos[1][i_id-1]
    # aa_C = pos[2][j_id-26]

    BC_patch_pos = (aa_B+aa_C)/2

    posB2 = BC_patch_pos - pos_COM[1]
    posC2 = BC_patch_pos - pos_COM[2]

    typB2 = ['Yb_%s'%i for i in range(len(posB2))]
    typC2 = ['Yc_%s'%i for i in range(len(posC2))]


    # THIRD INTERFACE AB [0,1], 20 aa interface MN Z

    data_AB = pd.read_csv('OPENSEQ/YIAM-YIAN', sep='\t')

    data_AB = data_AB.dropna()

    i_id = np.array([int(iid[:-2]) for iid in data_AB['i_id'].values])
    j_id = np.array([int(jid[:-2]) for jid in data_AB['j_id'].values])

    AB_scores = np.array(data_AB['p_sco'].values)
    AB_scores = AB_scores[:params.aa_interf[2]]

    # print("AB_scores",len(AB_scores))

    aa_A = pos[0][i_id[:params.aa_interf[2]]-1]
    aa_B = pos[1][j_id[:params.aa_interf[2]]-1]

    # aa_A = pos[0][i_id-1]
    # aa_B = pos[1][j_id-1]

    AB_patch_pos = (aa_A+aa_B)/2

    posA3 = AB_patch_pos - pos_COM[0]
    posB3 = AB_patch_pos - pos_COM[1]

    typA3 = ['Za_%s'%i for i in range(len(posA3))]
    typB3 = ['Zb_%s'%i for i in range(len(posB3))]



    ########

    positionsA = np.vstack((posA, posA1, posA3))
    positionsB = np.vstack((posB, posB2, posB3))
    positionsC = np.vstack((posC, posC1, posC2))

    positions = np.array([positionsA, positionsB, positionsC], dtype=object)

    typesA = np.concatenate((typA, typA1, typA3))
    typesB = np.concatenate((typB, typB2, typB3))
    typesC = np.concatenate((typC, typC1, typC2))

    types = np.array([typesA, typesB, typesC], dtype=object)

    massA = len(posA)
    massB = len(posB)
    massC = len(posC)

    mass = np.array([massA, massB, massC])

    evals_IA, evecs_IA = CalculatePrincipalMomentsOfInertia(posA)
    evals_IB, evecs_IB = CalculatePrincipalMomentsOfInertia(posB)
    evals_IC, evecs_IC = CalculatePrincipalMomentsOfInertia(posC)

    # evals_IA /= np.linalg.norm(evals_IA)
    # evals_IB /= np.linalg.norm(evals_IB)
    # evals_IC /= np.linalg.norm(evals_IC)

    # evals_IA = np.array([0,0,0])
    # evals_IB = np.array([0,0,0])
    # evals_IC = np.array([0,0,0])

    # evals_IA = np.array([1,1,1])
    # evals_IB = np.array([1,1,1])
    # evals_IC = np.array([1,1,1])

    evals_I = np.array([evals_IA, evals_IB, evals_IC])

    # print("evals A", evals_I[0])
    # print("evals B", evals_I[1])
    # print("evals C", evals_I[2])

    radiiA = np.array([1 for i in range(len(positionsA))])
    radiiB = np.array([1 for i in range(len(positionsB))])
    radiiC = np.array([1 for i in range(len(positionsC))])

    radii = np.array([radiiA, radiiB, radiiC], dtype=object)

    scores = np.array([AC_scores, BC_scores, AB_scores], dtype=object)

    return (positions, types, mass, evals_I, radii, scores, pos_COM)



def InitializeSystem_OPENSEQ(params):

    Nbbt = len(params.BBtypes)


    ### TRIMERS
    (p, t, m, I, r, s, pCOM) = InitBuildingBlock_OPENSEQ_YIA(params)
    t_flat = np.concatenate((t[0], t[1], t[2]))
    r_flat = np.concatenate((r[0], r[1], r[2]))

    print("MO X", s[0])
    print("NO Y", s[1])
    print("MN Z", s[2])

    en_interf_X = np.sum(s[0])
    en_interf_Y = np.sum(s[1])
    en_interf_Z = np.sum(s[2])
    en_tot = en_interf_X + en_interf_Y + en_interf_Z

    params.morse_D0_Z = (en_tot - en_interf_X*params.morse_D0_X - en_interf_Y*params.morse_D0_Y) / en_interf_Z

    if params.morse_D0_Z < 0:
        print("Error: DX too high!")
        exit()

    ### DIMERS
    # (p, t, m, I, r, s, pCOM) = InitBuildingBlock_OPENSEQ_PFL(params)
    # t_flat = np.concatenate((t[0], t[1]))
    # r_flat = np.concatenate((r[0], r[1]))

    # List of unique types, the corresponding indices
    t_unique, i_unique, typeids = np.unique(t_flat, return_index=True, return_inverse=True)

    # Empty list of radii
    rsum = []
    for i in range(len(r_flat)):
        rsum.append(r_flat[i])

    r_unique = [rsum[i] for i in i_unique]

    # Number of unique types
    n_t = len(t_unique)

    # Create and populate the particle types with type name and radius
    PartTypes = [ParticleType() for i in range(n_t)]
    for i in range(n_t):
        PartTypes[i].typeString = t_unique[i]
        PartTypes[i].radius = r_unique[i] # Create and populate the particle types with type name and radius

    # Create and populate the building block type: in this case only one
    BBlockTypeList = []

    count = 0

    for i in range(Nbbt):
        BBlockType = BuildingBlockType() # for i in range(n_bb)
        BBlockType.ID = i
        BBlockType.n = len(p[i])
        BBlockType.positions = np.array(p[i])
        BBlockType.typeids = typeids[count:count+BBlockType.n]
        BBlockType.mass = m[i]
        BBlockType.moment_inertia = I[i]

        BBlockTypeList.append(BBlockType)

        count += BBlockType.n

    # List of potentials
    Pots = [RepulsivePotential(), MorseXPotential(), MorseXRepulsivePotential()]

    # Create the interaction object
    Inter = Interactions()
    Inter.potentials = Pots
    Inter.InitializeMatrix(n_t)

    morse_rcut = 8.0/params.morse_a + params.morse_r0

    # non-interacting particles: N
    # first interface:  A0-C0
    # second interface: B1-C1
    # third interface:  A2-B2

    # Populate interaction matrix
    for i in range(n_t):
        for j in range(n_t):

            # if( t_unique[i] == t_unique[j] and t_unique[i] == 'N' ):
            #         # Repulsion between real spheres
            #         Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)

            if (len(t_unique[i]) == 1 and len(t_unique[j]) == 1):
                # Repulsion between real spheres
                # print("rep",i,t_unique[i],j,t_unique[j])
                Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)
            elif (t_unique[i][2:] == t_unique[j][2:] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == t_unique[j][0]):
                # Attraction between patches of all interfaces
                # print("att",t_unique[i],t_unique[j])
                if t_unique[i][0] =='X':
                    interf = 0
                    D0_interf = params.morse_D0_X
                elif t_unique[i][0] =='Y':
                    interf = 1
                    D0_interf = params.morse_D0_Y
                elif t_unique[i][0] =='Z':
                    interf = 2
                    D0_interf = params.morse_D0_Z
                # interf = int(t_unique[i][1]) # can be 0,1,2
                patch = int(t_unique[i][3:]) # one of the aa_interf[interf]
                score = s[interf][patch]     # the corresponding score of the bond
                Inter.matrix[i][j][1] = PotParam().SetMorseX(0, morse_rcut, params.morse_D0*D0_interf*score, params.morse_a, params.morse_r0, morse_rcut/2.)

            # for now let's try with no repulsion
            # elif( t_unique[i] == t_unique[j] and t_unique[i] != 'A' ):
            #         # Repulsion between patches of the same type (e.g. two G1's)
            #         Inter.matrix[i][j][2] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )


    # Create and populate the system definition object
    SystDef = SystemDefinition()
    SystDef.buildingBlockTypeList = BBlockTypeList
    SystDef.particleTypes = PartTypes
    SystDef.interactions = Inter
    SystDef.concentration = params.concentration
    SystDef.Lxyz = 10.0
    SystDef.kBT = params.kT_brown
    SystDef.seed = rdm.randint(1, 1001)
    SystDef.basename = params.fnbase
    SystDef.posCOM = pCOM

    return SystDef



def ConvertToMatrix(mi, euler_scheme):
    """
    Convert a set x,y,z,alpha,beta,gamma into a jts transformation matrix
    """
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5],axes=euler_scheme)
    return jnp.matmul(T, R)


def setup_system(SystDef):

    BBt = SystDef.buildingBlockTypeList
    bbCOM = SystDef.posCOM
    Nbbt = len(BBt)

    ref_ppos = []
    ref_id = []

    for bbt in BBt:
        ref_ppos.append(bbt.positions)
        ref_id.append(bbt.ID)


    #### specific for a fully connected trimer!!! ####

    ref_pposA = ref_ppos[0]
    ref_pposB = ref_ppos[1]
    ref_pposC = ref_ppos[2]

    ref_pposAB = [ref_pposA, ref_pposB]
    ref_pposBC = [ref_pposB, ref_pposC]
    ref_pposCA = [ref_pposC, ref_pposA]

    ref_pposABC = [ref_pposA, ref_pposB, ref_pposC]

    ref_ppos_tot = [[ref_pposA], [ref_pposB], [ref_pposC], ref_pposAB, ref_pposBC, ref_pposCA, ref_pposABC]

    bbA = jnp.array([ref_id[0]])
    bbB = jnp.array([ref_id[1]])
    bbC = jnp.array([ref_id[2]])

    bbAB = jnp.squeeze(jnp.vstack((bbA, bbB)))
    bbBC = jnp.squeeze(jnp.vstack((bbB, bbC)))
    bbCA = jnp.squeeze(jnp.vstack((bbC, bbA)))

    bbABC = jnp.squeeze(jnp.vstack((bbA, bbB, bbC)))

    bb_tot = [bbA, bbB, bbC, bbAB, bbBC, bbCA, bbABC]

    ref_q0A = jnp.array([bbCOM[0][0], bbCOM[0][1], bbCOM[0][2], 0, 0, 0], dtype=jnp.float64) + 1e-14
    ref_q0B = jnp.array([bbCOM[1][0], bbCOM[1][1], bbCOM[1][2], 0, 0, 0], dtype=jnp.float64) + 1e-14
    ref_q0C = jnp.array([bbCOM[2][0], bbCOM[2][1], bbCOM[2][2], 0, 0, 0], dtype=jnp.float64) + 1e-14

    ref_q0AB = jnp.concatenate((ref_q0A, ref_q0B))
    ref_q0BC = jnp.concatenate((ref_q0B, ref_q0C))
    ref_q0CA = jnp.concatenate((ref_q0C, ref_q0A))

    ref_q0ABC = jnp.concatenate((ref_q0A, ref_q0B, ref_q0C))

    ref_q0_tot = [ref_q0A, ref_q0B, ref_q0C, ref_q0AB, ref_q0BC, ref_q0CA, ref_q0ABC]



    #### specific for a dimer!!! ####
    """
    ref_pposA = ref_ppos[0]
    ref_pposB = ref_ppos[1]

    ref_pposAB = [ref_pposA, ref_pposB]

    ref_ppos_tot = [[ref_pposA], [ref_pposB], ref_pposAB]

    bbA = jnp.array([ref_id[0]])
    bbB = jnp.array([ref_id[1]])

    bbAB = jnp.squeeze(jnp.vstack((bbA, bbB)))

    bb_tot = [bbA, bbB, bbAB]

    ref_q0A = jnp.array([bbCOM[0][0], bbCOM[0][1], bbCOM[0][2], 0, 0, 0], dtype=jnp.float64) + 1e-14
    ref_q0B = jnp.array([bbCOM[1][0], bbCOM[1][1], bbCOM[1][2], 0, 0, 0], dtype=jnp.float64) + 1e-14

    ref_q0AB = jnp.concatenate((ref_q0A, ref_q0B))

    ref_q0_tot = [ref_q0A, ref_q0B, ref_q0AB]
    """




    def cluster_energy(q, euler_scheme, ppos, BBlocks):
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
            Mat.append(ConvertToMatrix(q[qi:qi+6], euler_scheme))

        # apply building block matrix to spheres positions
        real_ppos = []
        for i in range(Nbb):
            real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

        tot_energy = jnp.float64(0)

        if Nbb == 1:
            return tot_energy
        else:
            for bb1 in range(len(BBlocks)):
                for bb2 in range(bb1+1,len(BBlocks)):

                    # vector positions of bb1, bb2 (patches)
                    # make matrix of distances
                    # apply the energy function to the matrix with broadcasting

                    for p1, t1 in enumerate(BBt[BBlocks[bb1]].typeids):
                        for p2, t2 in enumerate(BBt[BBlocks[bb2]].typeids):
                            if Inter.matrix[t1][t2][1] != None: # and bb1!=bb2:
                                pos1 = real_ppos[bb1][p1]
                                pos2 = real_ppos[bb2][p2]
                                rmin  = Inter.matrix[t1][t2][1].rmin
                                rmax  = Inter.matrix[t1][t2][1].rmax
                                coeff = list(Inter.matrix[t1][t2][1].coeff.values())
                                r = jnp.linalg.norm(pos1-pos2)
                                ID1 = SystDef.particleTypes[t1]
                                ID2 = SystDef.particleTypes[t2]
                                tot_energy += Inter.potentials[1].E(r, rmin, rmax, coeff[0], coeff[1], coeff[2], coeff[3])

            return tot_energy

    return cluster_energy, ref_q0_tot, ref_ppos_tot, bb_tot


def add_variables(ma, mb, euler_scheme):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """
    Ma = ConvertToMatrix(ma, euler_scheme)
    Mb = ConvertToMatrix(mb, euler_scheme)
    Mab = jnp.matmul(Mb,Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans,angles))


def add_variables_all(mas, mbs, euler_scheme):
    """
    Given two vectors of length (6*n,), 'add' them
    per building block according to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0]//6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0]//6, 6))


    return jnp.reshape(vmap(add_variables, in_axes=(0, 0, None))(
        mas_temp, mbs_temp, euler_scheme), mas.shape)



def setup_variable_transformation(energy_fn, euler_scheme, q0, ppos, BBlocks):
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
    assert(Nbb*6==q0.shape[0])
    assert(len(ppos)==Nbb)
    assert(ppos[0].shape[1]==3)

    E = energy_fn(q0, euler_scheme, ppos, BBlocks)
    G = grad(energy_fn)(q0, euler_scheme, ppos, BBlocks)
    H = hessian(energy_fn)(q0, euler_scheme, ppos, BBlocks)

    print("Energy",E)
    # print("Gradient",G)

    evals, evecs = jnp.linalg.eigh(H)

    print("evals",evals)


    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(evals < zeromode_thresh, 1, 0))
    if Nbb == 1 or E == 0.0:
        zvib = 1.0
    else:
        zvib = jnp.product(jnp.sqrt(2.0*jnp.pi / (jnp.abs(evals[6:])+1e-12)))

    def ftilde(nu):
        return jnp.matmul(evecs.T[6:].T, nu[6:])

    def f_multimer(nu, addq0=True):
        dq_tilde = ftilde(nu)
        if not addq0:
            q_tilde = dq_tilde
        else:
            q_tilde = add_variables_all(q0, dq_tilde, euler_scheme)

        nu_bar_repeat = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return add_variables_all(q_tilde, nu_bar_repeat, euler_scheme)

    def f_monomer(nu, addq0=True):
        return nu

    if Nbb == 1:
        f = f_monomer
    else:
        f = f_multimer

    return f, num_zero_modes, zvib


def GetJmean_method1(f, key, Nbb, nrandom = 100000, euler_scheme='sxyz'):
    def random_euler_angles(key, euler_scheme):
        quat = jts.random_quaternion(None, key)
        return jnp.array(jts.euler_from_quaternion(quat, euler_scheme))

    def set_nu(angles):
        nu0=jnp.full((6*Nbb,), 0.0)
        return index_update(nu0, index[3:6], angles)

    def set_nu_random(key, euler_scheme):
        return set_nu(random_euler_angles(key, euler_scheme))

    key, *splits = random.split(key, nrandom+1)
    nus = vmap(set_nu_random, in_axes=(0, None))(jnp.array(splits), euler_scheme)

    Js = vmap(lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu, False))))(nus)

    mean = jnp.mean(Js)
    error = osp.stats.sem(Js)

    return mean, error

def Calculate_Zc(key, energy_fn, euler_scheme, ref_q0, ref_ppos, BBlocks, sigma, kBT, V):
    """
    Calculate Zc except without the lambdas
    """
    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, euler_scheme, ref_q0, ref_ppos, BBlocks)

    Js_mean, Js_error = GetJmean_method1(f, key, len(BBlocks))
    Jtilde = 8.0*(jnp.pi**2)*Js_mean

    E0 = energy_fn(ref_q0, euler_scheme, ref_ppos, BBlocks)
    boltzmann_weight = jnp.exp(-E0/kBT)
    print("boltzmann_weight", boltzmann_weight)
    print("Jtilde", Jtilde)
    print("zvib", zvib)
    return boltzmann_weight * V * (Jtilde/sigma) * zvib

def Calculate_yield_grancan(conc_list, Zc_list, Nc_list):
    Y_grancan = jnp.array([cl**Nc*Zc for (cl, Zc, Nc) in zip(conc_list, Zc_list, Nc_list)])
    return Y_grancan / jnp.sum(Y_grancan)


def Calculate_concentrations(trimer_Z, trimer_conc, V):
    conc_total_M = trimer_conc[0]
    conc_total_N = trimer_conc[1]
    conc_total_O = trimer_conc[2]

    for guess in range(3):
        if guess == 0:
            trimer_log_concs_guess = jnp.array(  # guess all stay as monomers
                [jnp.log(conc_total_M), jnp.log(conc_total_N), jnp.log(conc_total_O)] +
                [-1] * 4)

        elif guess == 1:
            trimer_log_concs_guess = jnp.array(  # guess MN dimer forms mostly and O stays as monomer
                [-1] * 2 + [jnp.log(conc_total_O)] + [jnp.log(conc_total_M)] + [-1] * 3)

        elif guess == 2:
            trimer_log_concs_guess = jnp.array(  # guess MNO trimer forms
                [-1] * 6 + [jnp.log(conc_total_M)])

        else:
            trimer_log_concs_guess = jnp.array(  # agnostic guess
                [jnp.log(conc_total_M), jnp.log(conc_total_N), jnp.log(conc_total_O)] +
                [jnp.log(conc_total_M), jnp.log(conc_total_N), jnp.log(conc_total_O)] +
                [jnp.log(conc_total_M)])

        #print(guess, "log conc guess", trimer_log_concs_guess)

        trimer_concs_sol = jnp.exp(fsolve(
            all_eqs, trimer_log_concs_guess, args=(
                conc_total_M, conc_total_N, conc_total_O, trimer_Z, V),
            # maxfev=(100*(7+1) * 20 ), xtol=1.49012e-08 / 20#, factor=0.1
        ))

        # Check that we were able to satisfy the equations
        trimer_concs_check = all_eqs(  # should be zero if the solution we found is a real solution
            jnp.log(trimer_concs_sol),
            conc_total_M, conc_total_N, conc_total_O, trimer_Z, V)

        print("trimer_concs_check",trimer_concs_check)

        if not jnp.all(jnp.isclose(trimer_concs_check, 0, atol=1e-10)):
            print('Was not able to satisfy all the equations simultaneously for guess # ' + str(guess))
        else:
            #break
            print("guess",guess)
            return trimer_concs_sol

    return trimer_concs_sol


def Full_Calculation(params, systdef):

    # setup_system returns the energy function and three lists corresponding to:
    # - ref_q0: 6N-dimensional COM coordinates of the BBs in the structure
    # - ref_ppos: the positions of the spheres of a building block wrt the bb COM
    # - bb: the ID of the building blocks in the cluster

    conc = systdef.concentration
    N = 3
    V = N / conc

    cluster_energy, ref_q0_list, ref_ppos_list, bb_list = setup_system(systdef)


    Nc = len(bb_list)
    key = random.PRNGKey(0)
    split = random.split(key, Nc)
    Zc_list = []

    for c in range(Nc):
        print("c", c)
        zc = Calculate_Zc(split[c],
                          cluster_energy,
                          'sxyz',
                          ref_q0_list[c],
                          ref_ppos_list[c],
                          bb_list[c],
                          sigma=params.sigma[c],
                          kBT=1.0,
                          V=N/conc)
        Zc_list.append(zc)


    conc_list = [conc for c in range(len(bb_list))]
    Nc_list = [len(bbs) for bbs in bb_list]

    print("conc_list", conc_list)
    print("Nc_list", Nc_list)
    print("Zc_list", Zc_list)


    pdb.set_trace()
    if Nc == 3:
        ZdimerOverZAZB = Zc_list[2] / (Zc_list[0]*Zc_list[1])

        yields = (1 + conc_list[0]*V*ZdimerOverZAZB + conc_list[1]*V*ZdimerOverZAZB - jnp.sqrt(4*conc_list[0]*V*ZdimerOverZAZB + (1+(-conc_list[0]+conc_list[1])*V*ZdimerOverZAZB)**2)) / (-1 + conc_list[0]*V*ZdimerOverZAZB + conc_list[1]*V*ZdimerOverZAZB + jnp.sqrt(4*conc_list[0]*V*ZdimerOverZAZB + (1 + (-conc_list[0] + conc_list[1])*V*ZdimerOverZAZB)**2))

    elif Nc == 7:
        conc_real = Calculate_concentrations(Zc_list, conc_list, V)
        yields = conc_real / jnp.sum(conc_real)

    yields_wr = Calculate_yield_grancan(conc_list, Zc_list, Nc_list)

    return yields, conc_real, yields_wr



if __name__ == "__main__":

    # python3 compute.py -i OPENSEQ/YIA_MNO_v3 -D 0.1 -c 0.00000001


    params = CParams()
    # params = CParams_dimer()


    # RunMinimizeEnergy_overlap
    # -i --inputname : input file
    # -D --morse_D0  : morse min
    # -a --morse_a   : morse strength
    # -c --conc      : concentration
    # -s --string    : output file
    # -t --testing   : simulation in testing mode
    parser = parserFunc()

    # list of the arguments of the parser
    args = parser.parse_args()

    # input file from which to read the BB aa positions
    # must be of type
    # aa_index x_coord y_coord z_coord
    params.inputfile = '%s' % (args.input_name)

    params.concentration = args.conc
    params.morse_D0 = args.morse_D0

    params.morse_D0_X = args.morse_D0_X
    params.morse_D0_Y = args.morse_D0_Y
    params.morse_D0_Z = args.morse_D0_Z

    # params.morse_a = args.morse_a

    # file where the trajectory is saved
    params.fnbase = 'temp_results/%s_D0%s_c%s' % (args.string, str(args.morse_D0), str(args.conc))

    # InitBuildingBlock_C3RK(params)
    # InitBuildingBlock_RK597(params)


    systdef = InitializeSystem_OPENSEQ(params)
    # systdef = InitializeSystem_dimers_tr(params)
    # systdef = InitializeSystem_C3RK(params)

    pdb.set_trace()

    Ys, Cs, Ywr = Full_Calculation(params, systdef)
    # Ys = Full_Calculation(systdef)
    # Ys, pc = Full_Calculation_can(params,systdef)

    for y, c, ywr in zip(Ys, Cs, Ywr):
        print(y, c, ywr)


    # print("\nYield",Ys)

    # if np.sum(Cs) < params.concentration:
    #     print("Solution not found")
    # else:
    #     print("So far so good")
