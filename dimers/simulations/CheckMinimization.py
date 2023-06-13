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
        N = [1,1]
        concentration = 0.001
        ls = 10.

        Ttot = 1e7      # actually total number of steps. Ttot_real=Ttot
        Trec_data = 10000
        Trec_traj = 10000
        dT = 0.0002

        rep_A = 500.0
        rep_alpha = 2.5

        morse_D0 = 3.0
        morse_a = 5.0

        morse_r0 = 0.

        kT_brown = 1.0
        seed_brown = random.randint(1,1001)

        fnbase = "test"

params = CParams()

hoomd.context.initialize("");

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

params.morse_D0 = args.morse_D0
params.morse_a = args.morse_a

params.fnbase = 'temp_results/%s_a%4.2f_D0%4.2f' % (args.string,args.morse_a,args.morse_D0)

SystDef = InitializeSystem_dimers(params)

PartTypes  = SystDef.particleTypes
BBlockTypeList = SystDef.buildingBlockTypeList

NbbTypes = len(BBlockTypeList)

SaveParamFile(params,SystDef.basename+"_params.txt")

PartTypes  = SystDef.particleTypes
BBlockTypeList = SystDef.buildingBlockTypeList

NbbTypes = len(BBlockTypeList)
C = ["C0","C1"]

snapshot = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=6), particle_types=C)

snapshot.particles.position[0] = [0, 0, 0]
snapshot.particles.position[1] = [0, 0, 2]
snapshot.particles.mass[0] = 1.0
snapshot.particles.mass[1] = 1.0
snapshot.particles.moment_inertia[0] = [2./5., 2./5., 2./5.]
snapshot.particles.moment_inertia[1] = [2./5., 2./5., 2./5.]

system = hoomd.init.read_snapshot(snapshot)

# Only unique types are listed
unique_types = SystDef.GetTypeStrings()

for ut in unique_types:
        system.particles.types.add(ut)

rigid = hoomd.md.constrain.rigid()

# # Populate the rigid body object
# for bbN in range(NbbTypes):
#         rigid.set_param(C[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

rigid.set_param('C0', types = ['A','X'], positions = [(0, 0, 0), (0, 0, 1)])
rigid.set_param('C1', types = ['A','Y'], positions = [(0, 0, 0), (0, 0, -1)])

# Creates the rigid body object and validate it
rigid.create_bodies()
rigid.validate_bodies()

# Creates the list to compute interactions
nl = hoomd.md.nlist.cell()

# Add the dumb center of mass particle to the particles list
alltypes = C+unique_types.tolist()

Inter = SystDef.interactions

# Number of potentials: in out case the Herzian and the Morse --> 2
num_potentials = len(Inter.potentials)

# Create the instance of the potentials in hoomd with the function table()
hoomd_potentials = [ hoomd.md.pair.table(width=1000,nlist=nl) for i in range(num_potentials) ]

# Initialize the potentials with the default params
for i in range(num_potentials):
        hoomd_potentials[i].pair_coeff.set(alltypes, alltypes,
                                           func  = Inter.potentials[i].E_f,
                                           rmin  = 0.,
                                           rmax  = 0.01,
                                           coeff = Inter.potentials[i].GetDefaultParams())

# Number of particle types
numPartTypes = len(PartTypes)

# Loop on all particles that can interact
for i in range(numPartTypes):
        for j in range(i,numPartTypes):
                for pi in range(num_potentials):
                        if (Inter.matrix[i][j][pi] != None):
                                hoomd_potentials[pi].pair_coeff.set(PartTypes[i].typeString, PartTypes[j].typeString,
                                                                    func  = Inter.potentials[pi].E_f,
                                                                    rmin  = Inter.matrix[i][j][pi].rmin,
                                                                    rmax  = Inter.matrix[i][j][pi].rmax,
                                                                    coeff = Inter.matrix[i][j][pi].coeff)



# Defines the integration scheme and the time step
hoomd.md.integrate.mode_standard(dt=params.dT);

# Groups particles
rigid = hoomd.group.rigid_center();

# Number of rigid bodies, check that it is correct
N = len(rigid)
#assert N==params.N

# Applies integration scheme to the group
hoomd.md.integrate.brownian(group=rigid, kT=SystDef.kBT, seed=SystDef.seed)

print("temperature",SystDef.kBT)

# Saves time, energy in a text file -- not really needed
hoomd.analyze.log(filename='%s_Data' % params.fnbase, quantities=['time','potential_energy'], period=params.Trec_data, header_prefix='#', overwrite=True, phase=0)

# Dump trajectories in a gsd file
hoomd.dump.gsd("%s_Traj.gsd" % params.fnbase,
               period=params.Trec_traj,
               group=hoomd.group.all(),
               overwrite=True,
               dynamic=['attribute']
       );

# Run like hell
hoomd.run(params.Ttot)


# I am happy
print("\n")
print("so far so good")
print("\n")


curr_snap = system.take_snapshot(all=True)

print(curr_snap.particles.position,curr_snap.particles.types)

snap_state = SystemState()

# populate the state corresponding to the entire snapshot
fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(params.N,curr_snap)
snap_state.positions = fullPos_com
snap_state.orientations = fullQtr_com
snap_state.bbTypes = fullTyp_com

print(snap_state.positions)




exit()





















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


BolFac, E, I, vib_entr = ComputeBoltzmannFactor(structureList,SystDef,params,SaveStructureFile=args.save_file)

Yield = BolFac/np.sum(BolFac)

SaveStructureDataList(structureList, listFilename)

for e,mi,ve,bf,y in zip(E,I,vib_entr,BolFac,Yield):
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





































"""
############## Use this to load an existing list ###############

listFilename = open('%slist.txt' % sys.argv[1],'r')

LoadStructureDataList(listFilename,structureList)

listFilename.close()

################################################################
"""

"""
TotFrames = len(t)

NumberOfWindows = 8

FramesPerWindow = floor(TotFrames/NumberOfwindows)

InitialFrame = TotFrames - FramesPerWindow*NumberOfWindows



# Simple example
# [0,1,2,3,4,5,6,7,8]

# [0]
# [1,2]
# [3,4]
# [5,6]
# [7,8]

# 4 NumberOfWindows
# 2 FramesPerWindow
# 1 InitialFrame

for window in range(NumberOfWindows): # 0, 1, 2, 3

    for frames in range(FramesPerWindow): # 0, 1

        # t[InitialFrame + window*FramesPerWindow + frames] --> t[1+0*2+0]=t[1], t[1+0*2+1]=t[2], t[1+1*2+0]=t[3], t[1+1*2+1]=t[4], t[1+2*2+0]=t[5], etc...

        curr_snap = t[InitialFrame + window*FramesPerWindow + frames]

        m = 0

        inter_matrix = PopulateMatrix(N,m,curr_snap,n1,n2,n3,th)

        AnalyzeAdjacencyMatrix(m,inter_matrix,structureList,curr_snap,n1,n3)

        print("snap %d of window %d out of %d windows" % (frame,window,NumberOfWindows))

    listFilename = open('%swin%d.txt' % (sys.argv[1],window),'w')

    SaveStructureDataList(structureList, listFilename)

    listFilename.close()




for cnt, curr_snap in enumerate(t):
"""

"""
curr_snap = t[len(t)-1]


# Populate the interaction matrix with the function PopulateMatrix(N,t)
# m=0 --> creates the building blocks interaction matrix
# m=1 --> creates the surfaces interaction matrix
# m=2 --> creates the columns interaction matrix
# m=3 --> creates the particles interaction matrix

m = 2

inter_matrix = PopulateMatrix(N,m,curr_snap,n1,n2,n3,th)

AnalyzeAdjacencyMatrix(m,inter_matrix,structureList,curr_snap,n1,n3)

#print("snap %d out of %d" % (cnt,len(t)))



listFilename = open('%slist.txt' % sys.argv[1],'w')

SaveStructureDataList(structureList, listFilename)

listFilename.close()


"""
