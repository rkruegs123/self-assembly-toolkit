"""/n/brenner_lab/Lab/baker_proteins/overlap_model

This code takes as input a gsd file corresponding to the output of
a previously run simulation. For each snapshot
1. the energy is minimized
2. the adjacency matrix is computed
3. the adjacency matrix for curr_snap i is compared with the on at curr_snap i-1
   3.1. if the same, break
   3.2. if different, find the connected submatrices, corresponding to the clusters
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
        N = [1000,1000]
        concentration = 0.001
        ls = 10.

        Ttot = 1e5      # actually total number of steps. Ttot_real=Ttot
        Trec_data = 10000
        Trec_traj = 10000
        dT = 0.0002

        rep_A = 500.0
        rep_alpha = 2.5

        morse_D0 = 3.0
        morse_a = 5.0

        morse_r0 = 0.

        kT_brown = 0.0
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

# t = gsd.hoomd.open(inputname,'rb')
# TotFrames = len(t)

inter_matrix = np.zeros( (params.N[0],params.N[1]), dtype=int)

structureList = [] #list of structureData objects

m = 0

listFilename = open('temp_results/%slist.txt' % args.input_name,'w')

for curr_snap in range(999,1000):

        # the output of the minimization full name
        #params.fnbase = 'temp_results/%s_c%4.6f_D0%4.2f_snap%d' % (args.string,args.conc,args.morse_D0,curr_snap)

        #params.fnbase = 'temp_results/%s_c%4.6f_D0%4.2f_snap%d' % (args.string,args.conc,args.morse_D0,curr_snap)

        params.fnbase = 'temp_results/%s_snap%d' % (args.input_name,curr_snap)

        SystDef = InitializeSystem_dimers(params)

        hoomd.context.initialize("");

        # the function RunEnergyMinimization is in RunMinimizeEnergy
        RunEnergyMinimization(params,SystDef,curr_snap,inputname)

        print(curr_snap)

        basename_i  = params.fnbase
        inputname_i = basename_i+'_EMin.gsd'

        # to analyze a gsd snapshot, use gsd.hoomd.open(), to initialize a
        # simulation from a snapshot, use hoomd.data.gsd_snapshot()
        min_traj_i = gsd.hoomd.open(inputname_i,'rb')

        len_traj = len(min_traj_i)

        # take the last snap pf the minimization
        snap_i = min_traj_i[len_traj-1]

        snap_state = SystemState()
        # populate the state corresponding to the entire snapshot
        fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(params.N,snap_i)
        snap_state.positions = fullPos_com
        snap_state.orientations = fullQtr_com
        snap_state.bbTypes = fullTyp_com

        # populate the adjacency matrix based on the interactions between particles
        #inter_matrix_i = PopulateMatrixSystDef(m,params.n1,params.n2,params.n3,params.th,snap_state,SystDef)
        inter_matrix_i = PopulateMatrixDimers(snap_state,SystDef)

        print("Snap",curr_snap,": Interaction matrix populated")

        # The next step will need to be removed/changed when we will
        # measure the yield from the simulations, because we then want the
        # counts from all snapshots to compute statistical averages

        # Analyze the adjacency matrix just created only if it is
        # different from the one created on the previous step
        if(np.array_equal(inter_matrix_i,inter_matrix)):
                continue
        else:
                inter_matrix = inter_matrix_i
                AnalyzeAdjacencyMatrix(params.N,params.concentration,m,inter_matrix,structureList,snap_i)
                print("Snap",curr_snap,": Adjacency matrix analyzed")


BolFac, E, I, vib_entr, vib_evals, vib_evecs = ComputeBoltzmannFactor(structureList,SystDef,params,SaveStructureFile=args.save_file)

Yield = BolFac/np.sum(BolFac)

SaveStructureDataList(structureList, listFilename)

for e,mi,evl,evc,ve,bf,y in zip(E,I,vib_evals,vib_evecs,vib_entr,BolFac,Yield):
        listFilename.write("\nEnergy %f\n" % e)
        listFilename.write("Inertia %f\n" % mi)
        listFilename.write("Eigenvalues ")
        listFilename.write(" ".join(str(elem) for elem in evl) + "\n")
        listFilename.write("Eigenvectors ")
        listFilename.write(" ".join(str(elem) for elem in evc) + "\n")
        listFilename.write("Vibration %f\n" % ve)
        listFilename.write("Boltzmann factor %f\n" % bf)
        listFilename.write("Yield %f\n\n" % y)

print("\nEnergy",E)
print("Inertia",I)
print("Eigenvalues",vib_evals)
print("Eigenvectors",vib_evecs)
print("Vibration",vib_entr)
print("Boltzmann factor",BolFac)
print("Yield",Yield)

listFilename.close()
