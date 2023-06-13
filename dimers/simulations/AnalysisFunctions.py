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
from classDefinitions import *
from dynamicalMatrix_4 import *
from RunFunctions import *
from RunMinimizeEnergy import *


def dist_lin(r1,r2,Lx,Ly,Lz):

    """
    Adjusts the linear distances to take care of the boundary conditions

    """

    dist_x = r1[0]-r2[0]
    if abs(dist_x) > Lx/2:
        if dist_x > 0:
            dist_x -= Lx
        else:
            dist_x += Lx

    dist_y = r1[1]-r2[1]
    if abs(dist_y) > Ly/2:
        if dist_y > 0:
            dist_y -= Ly
        else:
            dist_y += Ly

    dist_z = r1[2]-r2[2]
    if abs(dist_z) > Lz/2:
        if dist_z > 0:
            dist_z -= Lz
        else:
            dist_z += Lz

    return dist_x, dist_y, dist_z


def dist_tot(r1,r2,Lx,Ly,Lz):

    """
    Distance between particles with boundary conditions

    """

    dist_x,dist_y,dist_z = dist_lin(r1,r2,Lx,Ly,Lz)
    return math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)




"""
These functions take the indices of the particle of interest
(surface, column, sphere) of a building block l and convert them to
the indices of the adjacency matrix

"""

# j, l -> I
def getIndex_surface(j,l,n2):
    # j will equal 0 or n2-1
    if(j==0):
        jtemp=0
    elif(j==n2-1):
        jtemp=1
    else:
        assert False
    n2temp = 2
    I = jtemp + l*n2temp
    return I

# j, k, l -> I
def getIndex_columns(j,k,l,n2,n3):
    # j will equal 0 or n2-1
    if(j==0):
        jtemp=0
    elif(j==n2-1):
        jtemp=1
    else:
        assert False
    n2temp = 2
    I = k + jtemp*n3 + l*n2temp*n3
    return I

#i, j, k, l -> I
def getIndex_spheres(i,j,k,l,n1,n2,n3):
    #j will equal 0 or n2-1
    if(j==0):
        jtemp=0
    elif(j==n2-1):
        jtemp=1
    else:
        assert False
    n2temp = 2
    I = i + k*n1 + jtemp*n1*n3 + l*n1*n2temp*n3
    return I





def getPosTypLet_fromSnap(N,snap):

    """
    From a snapshot extracts the positions, types (those are numerical
    labels corresponding to a type) and the list of alphabetical types
    for all spheres in the simulation

    """

    Ntot = np.sum(N)

    pos = snap.particles.position     # Positions of all particles
    qtr = snap.particles.orientation  # Orientations of all particles
    typ = snap.particles.typeid       # Types of all particles

    types_lett = snap.particles.types # Names of particle types

    # Saves positions of the centers of mass (first N particles)
    pos_com = pos[0:Ntot]
    qtr_com = qtr[0:Ntot]
    typ_com = typ[0:Ntot]

    # Saves positions and types of particles excluding the first N which correspond to the COM
    pos = pos[Ntot:len(pos)]
    typ = typ[Ntot:len(typ)]

    # Splits the position and type arrays in N arrays, one for each rigid body
    pos = np.array_split(pos,Ntot)
    typ = np.array_split(typ,Ntot)

    return np.array(pos_com), np.array(qtr_com), np.array(typ_com), np.array(pos), np.array(typ), np.array(types_lett)





def CreateMatrix(N,m,n1,n3):

    """
    Creates the basic matrix with the connections within the single
    building blocks. Takes the number of building blocks and a variable
    m defining which type of adjacency matrix we will use.

    -/m = 0 considers two blocks connected if any two complementary spheres are touching

    -/m = 1 considers two surfaces connected if any two complemetary spheres are touching

    -/m = 2 considers two columns connected if any two complementary spheres are touching

    -/m = 3 considers two spheres connected if they are complementary and touching

    """

    if m==0:
        # creates an empty matrix to be populated with the connected building blocks
        inter_matrix = np.zeros((N,N), dtype=int)

    elif m==1:
        # populates the matrix taking into account connected surfaces (belonging to the same building block)
        surface_block = np.zeros((2,2), dtype=int)
        for i in range(2):
            surface_block[i][i] = i+1 # In this way the connectivity on the building block is unique
            if i%2==0:
                surface_block[i+1][i] = 1
                surface_block[i][i+1] = 1

        inter_matrix = block_diag(*([surface_block] * N))

    elif m==2:
        # populates the matrix taking into account connected columns (belonging to the same building block)
        columns_block = np.zeros((2*n3,2*n3), dtype=int)
        for i in range(2*n3):
            columns_block[i][i] = i+1 # In this way the connectivity on the building block is unique
            for k in range(2*n3-1):
                if i%(2*n3)==k:
                    for l in range(2*n3-k-1):
                        columns_block[i+1+l][i] = 1
                        columns_block[i][i+1+l] = 1

        inter_matrix = block_diag(*([columns_block] * N))

    elif m==3:
        # populates the matrix taking into account connected spheres (belonging to the same building block)
        spheres_block = np.zeros((2*n3*n1,2*n3*n1), dtype=int)
        for i in range(2*n3*n1):
            spheres_block[i][i] = i+1 # In this way the connectivity on the building block is unique
            for k in range(2*n3*n1-1):
                if i%(2*n3*n1)==k:
                    for l in range(2*n3*n1-k-1):
                        spheres_block[i+1+l][i] = 1
                        spheres_block[i][i+1+l] = 1

        inter_matrix = block_diag(*([spheres_block] * N))

    return inter_matrix






def PopulateMatrixSystDef(m,SystState,SystDef):

    """
    Populates the matrix with the connections between building blocks/surfaces/columns

    """

    Ntot = len(SystState.bbTypes)
    L = (Ntot/SystDef.concentration) ** (1./3.)

    inter_matrix = np.zeros((Ntot,Ntot), dtype=int)

    # particles that can attract are SystDef.buildingBlockTypeList[bbi].typeid
    BBlockList = SystDef.buildingBlockTypeList

    for bb1, bbT1 in enumerate(SystState.bbTypes):
        for bb2, bbT2 in enumerate(SystState.bbTypes):
            for p1, t1 in enumerate(BBlockList[bbT1].typeids):
                for p2, t2 in enumerate(BBlockList[bbT2].typeids):
                    if SystDef.interactions.matrix[t1][t2][1] != None and bb2 != bb1:
                        # positions of the interacting spheres in bb1
                        # and bb2 relative to the COM of bb1 and bb2 respectively
                        pos1 = BBlockList[bbT1].positions[p1]
                        pos2 = BBlockList[bbT2].positions[p2]

                        # positions rotated accordind to the current state
                        pos1 = SystDef.TransAndRotPart(SystState.positions[bb1],SystState.orientations[bb1],pos1)
                        pos2 = SystDef.TransAndRotPart(SystState.positions[bb2],SystState.orientations[bb2],pos2)
                        distance = dist_tot(pos1,pos2,L,L,L)

                        typeString1 = SystDef.particleTypes[t1].typeString
                        typeString2 = SystDef.particleTypes[t2].typeString

                        typeRadius1 = SystDef.particleTypes[t1].radius
                        typeRadius2 = SystDef.particleTypes[t2].radius

                        if typeString1=='X' and typeString2 =='Y' and distance < SystDef.interactions.matrix[t1][t2][1].rmax:

                            #bblocks:
                            if m==0:
                                inter_matrix[bb1,bb2]+=1
                                inter_matrix[bb2,bb1]+=1

                            #surface:
                            elif m==1:
                                # Extract the index for the surface adjacency matrix
                                I1 = getIndex_surface(j1,bb1,n2)
                                I2 = getIndex_surface(j2,bb2,n2)
                                inter_matrix[I1,I2]+=1
                                inter_matrix[I2,I1]+=1

                            #columns:
                            elif m==2:
                                # Extract the index for the columns adjacency matrix
                                I1 = getIndex_columns(j1,k,bb1,n2,n3)
                                I2 = getIndex_columns(j2,k,bb2,n2,n3)
                                inter_matrix[I1,I2]+=1
                                inter_matrix[I2,I1]+=1

                            #spheres:
                            elif m==3:
                                # Extract the index for the spheres adjacency matrix
                                I1 = getIndex_spheres(i,j1,k,bb1,n1,n2,n3)
                                I2 = getIndex_spheres(i,j2,k,bb2,n1,n2,n3)
                                inter_matrix[I1,I2]+=1
                                inter_matrix[I2,I1]+=1

    return inter_matrix






def PopulateMatrixDimers(SystState,SystDef):

    """
    Populates the matrix with the connections between dimer spheres

    """

    Ntot = len(SystState.bbTypes)
    L = (Ntot/SystDef.concentration) ** (1./3.)

    inter_matrix = np.zeros((Ntot,Ntot), dtype=int)

    # particles that can attract are SystDef.buildingBlockTypeList[bbi].typeid
    BBlockList = SystDef.buildingBlockTypeList

    for bb1, bbT1 in enumerate(SystState.bbTypes):
        for bb2, bbT2 in enumerate(SystState.bbTypes):
            for p1, t1 in enumerate(BBlockList[bbT1].typeids):
                for p2, t2 in enumerate(BBlockList[bbT2].typeids):
                    if SystDef.interactions.matrix[t1][t2][1] != None and bb2 != bb1:
                        # positions of the interacting spheres in bb1
                        # and bb2 relative to the COM of bb1 and bb2 respectively
                        pos1 = BBlockList[bbT1].positions[p1]
                        pos2 = BBlockList[bbT2].positions[p2]

                        # positions rotated accordind to the current state
                        pos1 = SystDef.TransAndRotPart(SystState.positions[bb1],SystState.orientations[bb1],pos1)
                        pos2 = SystDef.TransAndRotPart(SystState.positions[bb2],SystState.orientations[bb2],pos2)
                        distance = dist_tot(pos1,pos2,L,L,L)

                        typeString1 = SystDef.particleTypes[t1].typeString
                        typeString2 = SystDef.particleTypes[t2].typeString

                        typeRadius1 = SystDef.particleTypes[t1].radius
                        typeRadius2 = SystDef.particleTypes[t2].radius

                        if typeString1=='X' and typeString2 =='Y' and distance < SystDef.interactions.matrix[t1][t2][1].rmax:
                            inter_matrix[bb1,bb2]+=1
                            inter_matrix[bb2,bb1]+=1
                            break

                else:
                    continue
                break
            else:
                continue
        else:
            continue
        break


    return inter_matrix






def InitializeMatrixDimers_tr(N):

    """
    Initialize empty matrix for dimers with triangular patch

    """

    monomer_block = np.zeros((3,3), dtype=int)
    for i in range(3):
        monomer_block[i][i] = i+1 # In this way the connectivity on the building block is unique
        for k in range(2):
            if i%3==k:
                for l in range(3-k-1):
                    monomer_block[i+1+l][i] = 1
                    monomer_block[i][i+1+l] = 1

    return block_diag(*([monomer_block] * N))






def PopulateMatrixDimers_tr(SystState,SystDef):

    """
    Populates the matrix with the connections between dimers with triangula patches

    """

    Ntot = len(SystState.bbTypes)
    L = (Ntot/SystDef.concentration) ** (1./3.)

    inter_matrix = InitializeMatrixDimers_tr(Ntot)

    # particles that can attract are SystDef.buildingBlockTypeList[bbi].typeid
    BBlockList = SystDef.buildingBlockTypeList

    for bb1, bbT1 in enumerate(SystState.bbTypes): # bblock of reference
        for bb2, bbT2 in enumerate(SystState.bbTypes): # bblock that potentially interact with bb1
            for p1, t1 in enumerate(BBlockList[bbT1].typeids): # particles in bb1
                for p2, t2 in enumerate(BBlockList[bbT2].typeids): # particles in bb2
                    # make sure that particles t1 and t2 interact and do not belong to the same bblock
                    if SystDef.interactions.matrix[t1][t2][1] != None and bb2 != bb1:
                        # positions of the interacting spheres in bb1
                        # and bb2 relative to the COM of bb1 and bb2 respectively
                        pos1 = BBlockList[bbT1].positions[p1]
                        pos2 = BBlockList[bbT2].positions[p2]

                        # positions rotated accordind to the current state
                        pos1 = SystDef.TransAndRotPart(SystState.positions[bb1],SystState.orientations[bb1],pos1)
                        pos2 = SystDef.TransAndRotPart(SystState.positions[bb2],SystState.orientations[bb2],pos2)
                        distance = dist_tot(pos1,pos2,L,L,L)

                        typeString1 = SystDef.particleTypes[t1].typeString
                        typeString2 = SystDef.particleTypes[t2].typeString

                        typeRadius1 = SystDef.particleTypes[t1].radius
                        typeRadius2 = SystDef.particleTypes[t2].radius

                        if typeString1[0] == typeString2[0] and typeString1[1] != typeString2[1]:
                            if typeString1[0] == 'B' and distance < SystDef.interactions.matrix[t1][t2][1].rmax:
                                inter_matrix[bb1*3,bb2*3]+=1
                                inter_matrix[bb2*3,bb1*3]+=1
                            elif typeString1[0] == 'R' and distance < SystDef.interactions.matrix[t1][t2][1].rmax:
                                inter_matrix[bb1*3+1,bb2*3+1]+=1
                                inter_matrix[bb2*3+1,bb1*3+1]+=1
                            elif typeString1[0] == 'G' and distance < SystDef.interactions.matrix[t1][t2][1].rmax:
                                inter_matrix[bb1*3+2,bb2*3+2]+=1
                                inter_matrix[bb2*3+2,bb1*3+2]+=1

    return inter_matrix





"""
DEPRECATED

"""
# A function where, given a "cl" map:
#   - figure out which building blocks are involved
#   - returns a HOOMD snapshot of the current structure examined
def GetStructureSnapshot(N, m, cl, fullSnapshot, n1, n3):
    # Get positions, types, etc from the full snapshot
    fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(N,fullSnapshot)

    pp = len(fullPos[0])            # Number of particles per protein

    Lx = fullSnapshot.configuration.box[0]
    Ly = fullSnapshot.configuration.box[1]
    Lz = fullSnapshot.configuration.box[2]

    block_c =[]                     # List of blocks in the current connected structure

    if m==1:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//2 == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//2)

    elif m==2:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//(2*n3) == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//(2*n3))

    elif m==3:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//(2*n3*n1) == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//(2*n3*n1))

    Nbl = len(block_c)

    smallSnapshot = hoomd.data.make_snapshot(N=Nbl,
                                                box=hoomd.data.boxdim(L=Lx),
                                                particle_types=['C'],
                                            )

    for bl in range(Nbl):
        smallSnapshot.particles.position[bl] = fullPos_com[block_c[bl]]
        smallSnapshot.particles.orientation[bl] = fullQtr_com[block_c[bl]]
        smallSnapshot.particles.mass[bl] = pp
        evals, evecs = CalculatePrincipalMomentsOfInertia(fullPos[block_c[bl]])
        smallSnapshot.particles.moment_inertia[bl] = evals

    return smallSnapshot





def GetStructCOMPosQtrTyp_fromSnap(N, cl, fullSnapshot):

    """
    Get positions, types, etc from the full snapshot

    """

    fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(N,fullSnapshot)

    pp = len(fullPos[0])            # Number of particles per protein

    block_c =[]                     # List of blocks in the current connected structure

    for i in range(len(cl)):
        new_block = True
        for j in range(len(block_c)):
            if cl[i]//3 == block_c[j]:
                new_block = False
                break
        if new_block == True:
            block_c.append(cl[i]//3)

    Nbl = len(block_c)

    pos_c = []
    qtr_c = []
    typ_c = []

    for bl in range(Nbl):
        pos_c.append(fullPos_com[block_c[bl]])
        qtr_c.append(fullQtr_com[block_c[bl]])
        typ_c.append(fullTyp_com[block_c[bl]])

    typ_c = np.array(typ_c)

    return np.array(pos_c), np.array(qtr_c), np.array([ int(temp[1:]) for temp in types_lett[typ_c] ])



def GetStructCOMPosQtrTyp_fromState(N, cl, state):

    """
    Get positions, types, etc from system state

    """

    fullPos_com = state.positions
    fullQtr_com = state.orientations
    fullTyp_com = state.bbTypes

    block_c =[]                     # List of blocks in the current connected structure

    for i in range(len(cl)):
        new_block = True
        for j in range(len(block_c)):
            if cl[i]//3 == block_c[j]:
                new_block = False
                break
        if new_block == True:
            block_c.append(cl[i]//3)

    Nbl = len(block_c)

    pos_c = []
    qtr_c = []
    typ_c = []

    for bl in range(Nbl):
        pos_c.append(fullPos_com[block_c[bl]])
        qtr_c.append(fullQtr_com[block_c[bl]])
        typ_c.append(fullTyp_com[block_c[bl]])

    return np.array(pos_c), np.array(qtr_c), np.array(typ_c)



# Generates a random alphanumeric string of 6 characters, used to
# generate random names to save snapshot files
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

    """
    Generates a random alphanumeric string of 6 characters, used to
    generate random names to save snapshot files

    """

    return ''.join(random.choice(chars) for _ in range(size))




"""
DEPRECATED

"""
# Extracts the cluster as a snapshot and saves it as a .gsd file, then
# saves the filename to the structureData object

# Not used: the function GetStructureSnapshot doesn't work for now
def SaveStructureAsSnapshot(N, m, sd, cl, fullSnapshot, n1, n3):

    smallSnapshot = GetStructureSnapshot(N, m, cl, fullSnapshot, n1, n3)

    #Get a new unique filename (Generate random and check for uniqueness??)
    rd_name = id_generator()

    cl_l = len(cl)

    if m==0:
        Nbl = cl_l

    elif m==1:
        Nbl = cl_l//2

    elif m==2:
        Nbl = cl_l//(2*n3)

    elif m==3:
        Nbl = cl_l//(2*n3*n1)

    # Saves trajectories in a gsd file
    outname = "temp_results/snaps/snap%d_%s.gsd" % (Nbl,rd_name)

    system = hoomd.init.read_snapshot(smallSnapshot)

    hoomd.dump.gsd(outname,
                   period=None,
                   group=hoomd.group.all(),
                   #group=rigid,
                   overwrite=True,
                   time_step=0,
                   dynamic=['attribute']
               );

    sd.filenames.append(outname)





"""
DEPRECATED

"""
# This is the function used instead: just dump positions in an .xyz
# file.
def SaveStructureAsSnapshot(N,m,sd,cl,fullSnapshot,n1,n3):

    # Get positions, types, etc of all particles in a snapshot
    fullPos_com, fullQtr_com, fullTyp_com, fullPos, fullTyp, types_lett = getPosTypLet_fromSnap(N,fullSnapshot)

    pp = len(fullPos[0])            # Number of particles per protein

    Lx = fullSnapshot.configuration.box[0]
    Ly = fullSnapshot.configuration.box[1]
    Lz = fullSnapshot.configuration.box[2]

    box = fullSnapshot.configuration.box

    block_c =[]                     # List of blocks in the current connected structure

    if m==1:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//2 == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//2)

    elif m==2:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//(2*n3) == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//(2*n3))

    elif m==3:
        for i in range(len(cl)):
            new_block = True
            for j in range(len(block_c)):
                if cl[i]//(2*n3*n1) == block_c[j]:
                    new_block = False
                    break
            if new_block == True:
                block_c.append(cl[i]//(2*n3*n1))

    Nbl = len(block_c)

    pos_c = []
    typ_c = []

    for bl in range(Nbl):
        pos_c.append(fullPos_com[block_c[bl]])
        typ_c.append(fullTyp_com[block_c[bl]])

    for bl in range(Nbl):
        for ppl in range(pp):
            pos_c.append(fullPos[block_c[bl]][ppl])
            typ_c.append(fullTyp[block_c[bl]][ppl])



    # Get a new unique filename (Generate random and check for uniqueness??)
    rd_name = id_generator()

    cl_l = len(cl)

    if m==0:
        Nbl = cl_l

    elif m==1:
        Nbl = cl_l//2

    elif m==2:
        Nbl = cl_l//(2*n3)

    elif m==3:
        Nbl = cl_l//(2*n3*n1)


    outname = "results/snap%d_%s.xyz" % (Nbl,rd_name)

    Nsnap = len(pos_c)

    with open(outname,"w") as text_file:

        text_file.write("%s\n" % Nsnap)

        text_file.write("box ")
        for bb in box:
            text_file.write("%s " % bb)
        text_file.write("\n")

        for j in range(Nsnap):
            text_file.write(
                types_lett[typ_c[j]]
                +" %s %s %s \n" % (
                    pos_c[j][0],
                    pos_c[j][1],
                    pos_c[j][2])
            )


    sd.filenames.append(outname)






class structureData_old:

    """
    Class for the adjacency matrices and the corresponding info (count, filenames, etc...)

    """


    def __init__(self,adjM,count=0,filenames=[]):
        self.adjacencyMatrix = copy.deepcopy(adjM)
        self.graph = nx.from_numpy_matrix(self.adjacencyMatrix)
        self.count = count
        self.filenames = copy.deepcopy(filenames)

    def print(self):
        print("structureData: ", self.adjacencyMatrix, self.count, self.filenames)


class structureData:

    """
    Class for the adjacency matrices and the corresponding info (count, filenames, etc...)

    """

    def __init__(self,adjM,count,pos,qtr,BBt):
        self.adjacencyMatrix = copy.deepcopy(adjM)
        self.graph = nx.from_numpy_matrix(self.adjacencyMatrix)
        self.count = copy.deepcopy(count)
        self.positions = copy.deepcopy(pos)
        self.quaternions = copy.deepcopy(qtr)
        self.BBTypes = copy.deepcopy(BBt)
        self.filenames = "temp"

    def print(self):
        print("structureData:\n", self.adjacencyMatrix, self.count, self.positions, self.quaternions)



class singleStructure:

    """
    Class for a single structure, contains:
    (- indices = a list of the indices of the monomers in the structure)
    - positions = a list of the positions of the monomers in the structure
    - quaternions = a list of the orientations (quaternions) of the monomers in the structure

    """

    def __init__(self,pos,qtr):
        #self.indices = copy.deepcopy(ind)
        self.positions = copy.deepcopy(pos)
        self.quaternions = copy.deepcopy(qtr)




class structureType:

    """
    Class for a structure type, contains
    - adjacencyMatrix = adjacency matrix of the structure type
    - graph = graph version of the adjacency matrix
    - count = number of structures of this type found
    - BBTypes = building block types contained in the structure
    - filename = name of the file in which to store things if needed
    - singleStructureList = list of all single structures of this type

    """

    def __init__(self,adjM,count,BBt):
        self.adjacencyMatrix = copy.deepcopy(adjM)
        self.graph = nx.from_numpy_matrix(self.adjacencyMatrix)
        self.count = copy.deepcopy(count)
        self.BBTypes = copy.deepcopy(BBt)
        self.filenames = "temp"
        self.singleStructureList = []

    def print(self):
        print("\nstructureType:\n", "len graph", len(self.graph), "\nBBTypes", self.BBTypes, "\ncount", self.count, "\nadj matrix\n", self.adjacencyMatrix)






def structData2state(structData,state):

    """
    This function converts the structData resulting from the
    analysis of the adjacency matrix to a state object

    """

    graph = structData.graph

    # number of bblocks in the current structure
    bb_n = int(len(graph)/4)

    state.positions = structData.positions
    state.orientations = structData.quaternions
    state.bbTypes = structData.BBTypes





def structType2state(structType,j,state):

    """
    This function converts the structType resulting from the
    analysis of the adjacency matrix to a state object

    """

    state.positions = structType.singleStructureList[j].positions
    state.orientations = structType.singleStructureList[j].quaternions
    state.bbTypes = structType.BBTypes





def SaveStructureDataList(structureList, listFilename):

    """
    Save a list of structureData objects
    Each row corresponds to one structureData object, as follows:
    count n M[0,0],M[0,1],...,M[0,N];M[1,0],...,M[n,n] [filename1, ..., filenameN]

    """

    for sd in structureList:
        n_adjM = len(sd.adjacencyMatrix[0])
        listFilename.write("%d " % sd.count)
        listFilename.write("%d " % n_adjM)
        for i in range(n_adjM):
            for j in range(n_adjM):
                listFilename.write("%d," % sd.adjacencyMatrix[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(";")
        listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)

        listFilename.write(" ")
        for i in range(int(n_adjM)):
            for j in range(3):
                listFilename.write("%f," % sd.positions[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(";")
            for j in range(4):
                listFilename.write("%f," % sd.quaternions[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(" ")
        listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
        listFilename.write(" %s" % sd.filenames)
        listFilename.write("\n")




def SaveStructureTypeList(curr_snap, structureList, listFilename):

    """
    Save a list of structureType objects
    Each row corresponds to one structureData object, as follows:
    count n M[0,0],M[0,1],...,M[0,N];M[1,0],...,M[n,n] pos quat

    """

    listFilename.write("\nt=%d\n" % curr_snap)

    for st in structureList:
        n_adjM = len(st.adjacencyMatrix[0])

        listFilename.write("%d " % st.count)
        listFilename.write("%d " % n_adjM)
        for i in range(n_adjM):
            for j in range(n_adjM):
                listFilename.write("%d," % st.adjacencyMatrix[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(";")
        listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)

        listFilename.write(" ")
        for i in range(int(n_adjM/3)): # divide by 3 because each block is composed of 3 spheres and the adj matrix takes into account all interactions
            for j in range(3):
                listFilename.write("%f," % st.singleStructureList[0].positions[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(";")
            for j in range(4):
                listFilename.write("%f," % st.singleStructureList[0].quaternions[i][j])
            listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
            listFilename.write(" ")
        listFilename.seek(listFilename.tell() - 1, os.SEEK_SET)
        listFilename.write(" %s" % st.filenames)
        listFilename.write("\n")





"""
DEPRECATED

"""
# Load a list of structureData objects
#   - assumes the above data format
#   - creates a new list of structureData objects and returns the list
def LoadStructureDataList(listFilename,structureList):

    lines = listFilename.readlines()
    #filenames = []

    for x in lines:
        count=int(x.split(' ')[0])
        adjM=np.matrix(x.split(' ')[2])
        #filenames.append(x.split(' ')[3:])

        sd=structureData(adjM,count)
        structureList.append(sd)

    for sd in structureList:
        sd.print()




def CenterCluster(L,pos):

    """
    Sets the COM of a cluster to [0,0,0]

    """

    bbN = len(pos)
    for r1 in range(bbN):
        for r2 in range(r1+1,bbN):
            dist_x = pos[r2][0]-pos[r1][0]
            if dist_x > L/2:
                pos[r2][0] -= L
            elif dist_x < -L/2:
                pos[r2][0] += L

            dist_y = pos[r2][1]-pos[r1][1]
            if dist_y > L/2:
                pos[r2][1] -= L
            elif dist_y < -L/2:
                pos[r2][1] += L

            dist_z = pos[r2][2]-pos[r1][2]
            if dist_z > L/2:
                pos[r2][2] -= L
            elif dist_z < -L/2:
                pos[r2][2] += L

    com = np.array([0.,0.,0.])
    for p in pos:
        com += p
    com /= len(pos)

    for p in pos:
        p -= com

    com = np.array([0.,0.,0.])
    for p in pos:
        com += p
    com /= len(pos)

    return pos







def AnalyzeAdjacencyMatrix(N,conc,m,matrix,structureList,snap,saveSnapshots=True):

    """
    Takes a snapshot and classifies the structures observed

    """

    Ntot = np.sum(N)

    # compute the size of the simulation box from the number of
    # particles and the concentration
    L = (Ntot/conc) ** (1./3.)

    # binarize the adjacency matrix, taking care that the diagonal
    # remains unique (1,2,3,4,etc)
    n = matrix.shape[0]
    graph_matrix = np.zeros( (n,n), dtype=int)
    cutoff = 0.5
    for i in range(n):
        for j in range(n):
            # the off diagonal terms are binarized
            if i!=j:
                if matrix[i,j] > cutoff:
                    graph_matrix[i,j] = 1
                else:
                    graph_matrix[i,j] = 0
            # the diagonal terms remain the same
            else:
                graph_matrix[i,j] = matrix[i,j]

    # create the graph corresponding to the adjacency matrix for the whole system
    G = nx.from_numpy_matrix(graph_matrix)

    # loop on all structures found in snap
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        cl = sorted(list(c))

        # number of bblocks in the current structure cl
        bb_n = int(len(cl))

        # take the current connected subgraph found and convert to adj
        # matrix
        mat = np.zeros((len(cl),len(cl)), dtype=int)
        for i in range(len(cl)):
            for j in range(len(cl)):
                mat[i,j]=graph_matrix[cl[i],cl[j]]

        # convert adj matrix to graph for single structure
        Gcluster = nx.from_numpy_matrix(mat)

        # check if cluster just found already exists
        found_cluster = False
        for struct in structureList:
            if(nx.is_isomorphic(Gcluster,struct.graph)):
                # if it exists, add 1 count to the corresponding
                # struct entry
                struct.count += 1
                found_cluster = True

        # if instead the cluster does not exist, create an entry with
        # the corresponding adj matrix and append it to the list
        if(found_cluster == False):
            pos_c, qtr_c, typ_c = GetStructCOMPosQtrTyp_fromSnap(N,cl,snap)

            pos_c = CenterCluster(L,pos_c)
            count = 1
            sd = structureData(mat,count,pos_c,qtr_c,typ_c)
            structureList.append(sd)

    for sd in structureList:
        sd.print()





def AnalyzeAdjacencyMatrix_new(N,conc,m,matrix,structureList,snap,saveSnapshots=True):

    """
    Takes a snapshot and classifies the structures observed.
    The difference with the previous function is that a new structure is created
    everytime and appended in the list of structures of the same type.

    """

    Ntot = np.sum(N)

    # compute the size of the simulation box from the number of
    # particles and the concentration
    L = (Ntot/conc) ** (1./3.)

    # binarize the adjacency matrix, taking care that the diagonal
    # remains unique (1,2,3,4,etc)
    n = matrix.shape[0]
    graph_matrix = np.zeros( (n,n), dtype=int)
    cutoff = 0.5
    for i in range(n):
        for j in range(n):
            # the off diagonal terms are binarized
            if i!=j:
                if matrix[i,j] > cutoff:
                    graph_matrix[i,j] = 1
                else:
                    graph_matrix[i,j] = 0
            # the diagonal terms remain the same
            else:
                graph_matrix[i,j] = matrix[i,j]

    # create the graph corresponding to the adjacency matrix for the whole system
    G = nx.from_numpy_matrix(graph_matrix)

    # loop on all structures found in snap
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        cl = sorted(list(c))

        # number of bblocks in the current structure cl
        bb_n = int(len(cl))

        # take the current connected subgraph found and convert to adj
        # matrix
        mat = np.zeros((len(cl),len(cl)), dtype=int)
        for i in range(len(cl)):
            for j in range(len(cl)):
                mat[i,j]=graph_matrix[cl[i],cl[j]]

        # convert adj matrix to graph for single structure
        Gcluster = nx.from_numpy_matrix(mat)

        # get positions, quaternions and types for the current structure
        pos_c, qtr_c, typ_c = GetStructCOMPosQtrTyp_fromSnap(N,cl,snap)
        pos_c = CenterCluster(L,pos_c)

        # istantiate a singleStructure object for the current
        # structure with the correct indices, positions and
        # quaternions
        sStruct = singleStructure(pos_c,qtr_c)

        # check if cluster just found already exists
        found_cluster = False
        for struct in structureList:
            if(nx.is_isomorphic(Gcluster,struct.graph)):
                # if it exists, add 1 count to the corresponding
                # struct entry
                struct.count += 1
                struct.singleStructureList.append(sStruct)
                found_cluster = True

        # if instead the cluster does not exist, create an entry with
        # the corresponding adj matrix and append it to the list
        if(found_cluster == False):
            pos_c, qtr_c, typ_c = GetStructCOMPosQtrTyp_fromSnap(N,cl,snap)

            pos_c = CenterCluster(L,pos_c)
            count = 1
            sd = structureType(mat,count,typ_c)
            sd.singleStructureList.append(sStruct)
            structureList.append(sd)

    for sd in structureList:
        sd.print()




def AnalyzeAdjacencyMatrix_state(N,conc,m,matrix,structureList,state,saveSnapshots=True):

    """
    Takes a state and classifies the structures observed

    """

    Ntot = np.sum(N)

    # compute the size of the simulation box from the number of
    # particles and the concentration
    L = (Ntot/conc) ** (1./3.)

    # binarize the adjacency matrix, taking care that the diagonal
    # remains unique (1,2,3,4,etc)
    n = matrix.shape[0]
    graph_matrix = np.zeros( (n,n), dtype=int)
    cutoff = 0.5
    for i in range(n):
        for j in range(n):
            # the off diagonal terms are binarized
            if i!=j:
                if matrix[i,j] > cutoff:
                    graph_matrix[i,j] = 1
                else:
                    graph_matrix[i,j] = 0
            # the diagonal terms remain the same
            else:
                graph_matrix[i,j] = matrix[i,j]

    # create the graph corresponding to the adjacency matrix for the whole system
    G = nx.from_numpy_matrix(graph_matrix)

    # loop on all structures found in snap
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        cl = sorted(list(c))

        # number of bblocks in the current structure cl
        bb_n = int(len(cl))

        # take the current connected subgraph found and convert to adj
        # matrix
        mat = np.zeros((len(cl),len(cl)), dtype=int)
        for i in range(len(cl)):
            for j in range(len(cl)):
                mat[i,j]=graph_matrix[cl[i],cl[j]]

        # convert adj matrix to graph for single structure
        Gcluster = nx.from_numpy_matrix(mat)

        # get positions, quaternions and types for the current structure
        pos_c , qtr_c, typ_c = GetStructCOMPosQtrTyp_fromState(N,cl,state)

        sStruct = singleStructure(pos_c,qtr_c)

        # check if cluster just found already exists
        found_cluster = False
        for struct in structureList:
            if(nx.is_isomorphic(Gcluster,struct.graph)):
                # if it exists, add 1 count to the corresponding
                # struct entry
                struct.count += 1
                struct.singleStructureList.append(sStruct)
                found_cluster = True

        # if instead the cluster does not exist, create an entry with
        # the corresponding adj matrix and append it to the list
        if(found_cluster == False):
            pos_c, qtr_c, typ_c = GetStructCOMPosQtrTyp_fromState(N,cl,state)
            count = 1
            sd = structureType(mat,count,typ_c)
            sd.singleStructureList.append(sStruct)
            structureList.append(sd)





def CountClusters(inputname,params,listFilename):

    """
    Main function: runs a loop over the snapshots and counts the clusters in each snap

    """

    # to analyze a gsd snapshot, use gsd.hoomd.open(), to initialize a
    # simulation from a snapshot, use hoomd.data.gsd_snapshot()
    full_traj = gsd.hoomd.open(inputname,'rb')
    len_traj = len(full_traj)


    # initialize matrix
    inter_matrix = InitializeMatrixDimers_tr(np.sum(params.N))

    # initialize strusture lists
    structureList = [] #list of structureData objects
    structureListMinimized = [] #list of structureData objects

    m = 0

    indices = []

    for ind in range(1 ,101):
        indices.append(int(ind*len_traj/101-1))

    #for curr_snap in range( len_traj-1, len_traj ):
    for cs_i,curr_snap in enumerate(indices):

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
        inter_matrix_i = PopulateMatrixDimers_tr(snap_state,SystDef)

        print("\nSnap",curr_snap,": Interaction matrix populated")

        # Analyze the adjacency matrix just created only if it is
        # different from the one created on the previous step
        if(np.array_equal(inter_matrix_i,inter_matrix)):
            print("Interaction matrix at",curr_snap,"identical to interaction matrix at",indices[cs_i-1])
            SaveStructureTypeList(curr_snap, structureListMinimized, listFilename)
            listFilename.write("Interaction matrix at t=%d identical to interaction matrix at t=%d\n" % (curr_snap,indices[cs_i-1]))
            listFilename.flush()
            continue
        else:
            inter_matrix = inter_matrix_i



        # initialize list counts to 0, it gets re-computed each time in the loop
        for i_sl in range(len(structureList)):
            structureList[i_sl].count = 0
        for i_slm in range(len(structureListMinimized)):
            structureListMinimized[i_slm].count = 0


        # analyze adjacency matrix of the full snapshot
        AnalyzeAdjacencyMatrix_new(params.N,params.concentration,m,inter_matrix,structureList,snap)

        print("\nSnap",curr_snap,": Adjacency matrix analyzed")


        # loop over the different structures types (monomer, dimer, trimer, etc)
        for i, ST in enumerate(structureList):

            # there are j_max structures of type i in the current snapshot
            j_max = ST.count

            # create a state object for each structure of type i
            state = [SystemState() for _ in range(j_max)]

            for j in range(j_max):

                # populate the state for structure j of type i
                structType2state(ST,j,state[j])

                # if not a monomer, minimize the energy
                #if len(ST.BBTypes) > 1:
                    #FireSingleEnMin_new(params,SystDef,state[j],j)

                # construct the adjacency matrix of the current state that needs to be analyzed
                int_mat_j_min = PopulateMatrixDimers_tr(state[j],SystDef)

                # analyzed the adjacency matrix of the current state, containing only one structure j of type i
                AnalyzeAdjacencyMatrix_state(ST.BBTypes,params.concentration,m,int_mat_j_min,structureListMinimized,state[j])

                print("Snap",curr_snap,": Minimized Adjacency matrix analyzed")

        # save the minimized structures in the file
        SaveStructureTypeList(curr_snap, structureListMinimized, listFilename)
        listFilename.flush()





"""
DEPRECATED

"""

def ComputeBoltzmannFactor(structureList,SystDef,params,SaveStructureFile=True):

    """
    Compute Boltzmann Factor - or partition function  of the single structure type

    """

    vib_entr = [ 1 for _ in range(len(structureList)) ]
    I        = [ 1 for _ in range(len(structureList)) ]
    E        = [ 0 for _ in range(len(structureList)) ]

    evals = []
    evecs = []

    BolFac   = [ 0 for _ in range(len(structureList)) ]

    for i, SD in enumerate(structureList):
        state = SystemState()
        structData2state(SD,state)

        N_i = len(state.bbTypes)

        if SaveStructureFile:
            rd_name = id_generator()
            outname = "temp_results/snaps/snap_%s.xyz" % (rd_name)
            outnameCOM = "temp_results/snaps/snapCOM_%s.xyz" % (rd_name)

            SystDef.SaveStateToXYZ_Full(outname,state)
            SystDef.SaveStateToXYZ_COMParticles(outnameCOM,state)

            SD.filenames = rd_name

        # Compute Hessian and dynamical matrix
        H, dynMat, E[i] = mainDynamicalMatrixCalc(SystDef,state)

        print("Etot",E[i])

        # Diagonalize dynamical matrix
        evals_DM, evecs_DM = np.linalg.eigh(dynMat)

        evals.append(evals_DM)
        evecs.append(evecs_DM)

        size_DM = len(evals_DM)

        # Compute vibrational entropy
        if size_DM > 6:
            # exclude the first 7 eigenvalues: 3 trans, 3 rot, 1 zero
            for ev in range(7,size_DM):
                if evals_DM[ev] > 1e-10:
                    vib_entr[i] *= np.sqrt(2*np.pi/evals_DM[ev])

        evals_I, evecs_I = CalculatePrincipalMomentsOfInertia(state.positions)

        for l in range(len(evals_I)):
            if evals_I[l] < 1e-7:
                evals_I[l] = 2./5.

        I[i] = np.prod(evals_I)

        BolFac[i] = np.exp( - (E[i] - np.log(SystDef.concentration)*N_i ) ) * np.sqrt(abs(I[i])) / N_i * vib_entr[i]

    return BolFac, E, I, vib_entr, evals, evecs



"""
DEPRECATED

"""

def ComputeAverageBoltzmannFactor(structureList,SystDef,params,SaveStructureFile=True):

    """
    Computes the Boltzmann factor of a structure averaged over all the structures found of that type

    """

    vib_entr = [ 1 for _ in range(len(structureList)) ]
    I        = [ 1 for _ in range(len(structureList)) ]
    E        = [ 0 for _ in range(len(structureList)) ]

    BolFac   = [ 0 for _ in range(len(structureList)) ]

    for i, ST in enumerate(structureList):

        N_i = len(ST.BBTypes)

        ev_DM = [ 0 for _ in range(N_i*6)]

        j_max = ST.count

        state      = [SystemState() for _ in range(j_max)]
        H_i        = [0 for _ in range(j_max)]
        dynMat_i   = [0 for _ in range(j_max)]
        E_i        = [0 for _ in range(j_max)]
        vib_entr_i = [1 for _ in range(j_max)]
        I_i        = [0 for _ in range(j_max)]

        ev_DM_i    = [[0 for _ in range(N_i*6)] for _ in range(j_max)]

        j = 0

        while j < j_max:
            structType2state(ST,j,state[j])

            COM_pos = state[j].positions
            COM_quat = state[j].orientations

            bb = state[j].bbTypes
            part_pos = [ SystDef.buildingBlockTypeList[b].positions for b in bb ]
            typeids = [ SystDef.buildingBlockTypeList[b].typeids for b in bb ]

            rpos = []

            if N_i==1:
                evals_I = np.array([1.,1.,1.])
            if N_i==2:
                evals_I = np.array([2.,2.,1.])

            print("evals_I",evals_I)

            I_i[j] = np.prod(evals_I)

            # Compute Hessian and dynamical matrix
            H_i[j], dynMat_i[j], E_i[j] = mainDynamicalMatrixCalc(SystDef,state[j])

            # Diagonalize dynamical matrix
            ev_DM_i[j], evecs_DM = np.linalg.eigh(dynMat_i[j])

            #size_DM = len(evals_DM)
            size_DM = len(ev_DM_i[j])
            print("ev_DM",ev_DM_i[j])

            # Compute vibrational entropy
            if size_DM > 6:
                # exclude the first 6 eigenvalues: 3 trans, 3 rot
                for ev in range(6,size_DM):
                    vib_entr_i[j] *= np.sqrt(2*np.pi/ev_DM_i[j][ev])

            print("vib_entr_i_j",vib_entr_i[j])

            j+=1

        for ev in range(N_i*6):
            ev_DM[ev] = np.average(np.transpose(ev_DM_i)[ev])
            print("mean ev",ev,ev_DM[ev])
            print("stdv ev",ev,np.std(np.transpose(ev_DM_i)[ev]))

        E[i] = np.average(E_i)
        I[i] = np.average(I_i)
        vib_entr[i] = np.average(vib_entr_i)

        print("vibr entropy mean", vib_entr[i])
        print("mom inertia mean", I[i])
        print("vibr standard dev", np.std(vib_entr_i))
        print("inertia stand dev", np.std(I_i))

        BolFac[i] = np.exp( - (E[i] - np.log(SystDef.concentration)*N_i ) ) * np.sqrt(abs(I[i])) * vib_entr[i]

    return BolFac, E, I, vib_entr
