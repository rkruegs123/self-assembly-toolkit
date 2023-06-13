"""/n/brenner_lab/Lab/baker_proteins/overlap_model

This code takes as input a gsd file corresponding to the output of
a previously run simulation. For each snapshot
1. the adjacency matrix is computed
2. the adjacency matrix for curr_snap i is compared with the on at curr_snap i-1
   2.1. if the same, break
   2.2. if different, find the connected submatrices, corresponding to the clusters
3. the energy for a given structure is minimized
The newly found structures are added to a list that contains
   - adjacency matrix
   - inertia tensor --> NO, computed for the positions afterwards
   - positions of building blocks
   - orientations of building blocks
The list is then used to compute the yield for each structure.

"""

from __future__ import division
import sys
import gsd.fl
import gsd.hoomd
import hoomd
import hoomd.md
import math
import numpy as np

import networkx as nx
import string
import random
from scipy.linalg import block_diag
import os
import copy
from AnalysisFunctions import *
from RunMinimizeEnergy import *

from RunFunctions import *
from dynamicalMatrix_4 import *



# Parameters
class CParams:
        # these are all class variables, they do not change in
        # different instances of the class CParams
        N = [9,9]
        concentration = 0.001
        ls = 10.

        Ttot = 1e5      # actually total number of steps. Ttot_real=Ttot
        Trec_data = 10000
        Trec_traj = 10000
        dT = 0.0002

        rep_A = 500.0
        rep_alpha = 2.5

        morse_D0 = 7.0
        morse_D0_r = 1.0 # 1.0
        morse_D0_g = 1.0 # 0.5
        morse_D0_b = 1.0 # 3.0

        morse_a = 5.0

        morse_r0 = 0.

        kT_brown = 1.0
        seed_brown = random.randint(1,1001)

        fnbase = "test"

params = CParams()





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

# if in testing mode, change the following params
if(args.testing):
        params.N = [27,27]
        params.Ttot = 1e5
        params.Trec_data = 10000
        params.Trec_traj = 10000

params.morse_D0 = args.morse_D0
params.concentration = args.conc
params.morse_a = args.morse_a

# input name
basename  = 'temp_results/%s_Traj' % args.input_name #sys.argv[1]
inputname = basename+'.gsd'

# to analyze a gsd snapshot, use gsd.hoomd.open(), to initialize a
# simulation from a snapshot, use hoomd.data.gsd_snapshot()
full_traj = gsd.hoomd.open(inputname,'rb')
len_traj = len(full_traj)


# initialize matrix
monomer_block = np.zeros((3,3), dtype=int)
for i in range(3):
        monomer_block[i][i] = i+1 # In this way the connectivity on the building block is unique
        for k in range(2):
                if i%3==k:
                        for l in range(3-k-1):
                                monomer_block[i+1+l][i] = 1
                                monomer_block[i][i+1+l] = 1

inter_matrix = block_diag(*([monomer_block] * params.N[0]))

# inter_matrix = np.zeros( (params.N[0]*3,params.N[1]*3), dtype=int)

structureList = [] #list of structureData objects
structureListMinimized = [] #list of structureData objects

m = 0

listFilename = open('temp_results/%slist.txt' % args.input_name,'w')

#indices = [int(len_traj/8-1), int(len_traj/4-1), int(len_traj/2-1), int(len_traj*3/4-1), int(len_traj*7/8-1), len_traj-1]

indices = []

for ind in range(1 ,11):
       indices.append(int(ind*len_traj/10-1))

#for curr_snap in range( len_traj-1, len_traj ):
for cs_i,curr_snap in enumerate(indices):

        # base filename of all files created from this snapshot
        params.fnbase = 'temp_results/%s_snap%d' % (args.input_name,curr_snap)

        #snap = full_traj[len_traj-1]
        snap = full_traj[curr_snap]

        # system definition for dimers
        SystDef = InitializeSystem_dimers_tr(params)

        # system state of the current snapshot
        snap_state = SystemState()
        # populate the state corresponding to the entire snapshot
        fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(params.N,snap)
        snap_state.positions = fullPos_com
        snap_state.orientations = fullQtr_com
        snap_state.bbTypes = fullTyp_com

        # populate the adjacency matrix based on the interactions between particles
        #inter_matrix_i = PopulateMatrixSystDef(m,params.n1,params.n2,params.n3,params.th,snap_state,SystDef)
        inter_matrix_i = PopulateMatrixDimers_tr(snap_state,SystDef)

        print("Snap",curr_snap,": Interaction matrix populated")

        #for i in range(len(structureList)):
        #        structureList[i].count = 0

        # The next step will need to be removed/changed when we will
        # measure the yield from the simulations, because we then want the
        # counts from all snapshots to compute statistical averages

        inter_matrix = inter_matrix_i
        for i_sl in range(len(structureList)):
                structureList[i_sl].count = 0
        for i_slm in range(len(structureListMinimized)):
                structureListMinimized[i_slm].count = 0

        # Analyze the adjacency matrix just created only if it is
        # different from the one created on the previous step
        #if(np.array_equal(inter_matrix_i,inter_matrix)):
        #        continue
        #else:
        #        inter_matrix = inter_matrix_i
        AnalyzeAdjacencyMatrix_new(params.N,params.concentration,m,inter_matrix,structureList,snap)
        print("Snap",curr_snap,": Adjacency matrix analyzed")


        for i, ST in enumerate(structureList):

                j_max = ST.count
                state = [SystemState() for _ in range(j_max)]
                for j in range(j_max):

                        structType2state(ST,j,state[j])

                        print("j",j)

                        if len(ST.BBTypes) > 1:
                                FireSingleEnMin(params,SystDef,state[j],j)
                                #GDEnMin(SystDef,state[j])
                                #ScipyEnMin(SystDef,state[j])

                                #ST.singleStructureList[j].positions = state[j].positions
                                #ST.singleStructureList[j].quaternions = state[j].orientations

                        int_mat_j_min = PopulateMatrixDimers_tr(state[j],SystDef)
                        AnalyzeAdjacencyMatrix_state(ST.BBTypes,params.concentration,m,int_mat_j_min,structureListMinimized,state[j])

                        print("Snap",curr_snap,": Minimized Adjacency matrix analyzed")

                        # print("ST.BBTypes",ST.BBTypes)

        SaveStructureTypeList(curr_snap, structureListMinimized, listFilename)
        listFilename.flush()

BolFac, E, I, vib_entr = ComputeAverageBoltzmannFactor(structureListMinimized,SystDef,params,SaveStructureFile=args.save_file)
#BolFac, E, vib_entr = ComputeAverageBoltzmannFactor(structureList,SystDef,params,SaveStructureFile=args.save_file)

Yield = BolFac/np.sum(BolFac)

#SaveStructureTypeList(999, structureList, listFilename)
#listFilename.flush()

for e,mi,ve,bf,y in zip(E,I,vib_entr,BolFac,Yield):
#for e,ve,bf,y in zip(E,vib_entr,BolFac,Yield):
        listFilename.write("\nEnergy %f\n" % e)
        listFilename.write("Inertia %f\n" % mi)
        listFilename.write("Vibration %f\n" % ve)
        listFilename.write("Boltzmann factor %f\n" % bf)
        listFilename.write("Yield %f\n\n" % y)

print("\nEnergy",E)
print("Inertia",I)
print("Vibration",vib_entr)
print("Boltzmann factor",BolFac)
print("Yield",Yield)

listFilename.close()
