import numpy as np
from potentials import *

#import transformations
import transformations
import potentials


#note: particle types are something that should be defined globally rather than "belonging" to an individual BuildingBlockType. Therefore, we will keep this separate


"""
ParticleType class
parameters:
        typeString -- string that uniquely identifies the type
	radius -- float that sets the radius of the particle (note: this is only for the repulsive interaction)
"""
class ParticleType:
        def __init__(self, typeString='AA', radius=1.0):
                self.typeString = typeString
                self.radius = radius

        def __str__(self):
                return "PType: "+self.typeString+" "+str(self.radius)
        def __repr__(self):
                return "PType: "+self.typeString+" "+str(self.radius)


"""
BuildingBlockType class
parameters:
	n -- number of particles in building block
	positions -- matrix with shape (n,DIM). position of every particle within bulding block
	typeids -- array with length n. There will be a separate list of Particle types (called particleTypes) that lists the different possible particle types. Therefore, the string that signifies the type of particle i is particleTypes[self.typeids[i]].typeString
	mass -- float. total mass of the building block
	moment_inertia -- moment of inertia for the entire building block. Currently this is a list of 3 numbers, should it be a matrix or does the building block have to be orientated along its principle axes?

"""
class BuildingBlockType:
        def __init__(self):
                self.n = 0
                self.positions = np.array([])
                self.typeids = np.array([])
                self.mass = 1.0
                self.moment_inertia = np.array([])







class PotParam:
	def __init__(self, rmin=0, rmax=0, coeff=dict()):
		self.rmin = rmin
		self.rmax = rmax
		self.coeff = coeff

	def __str__(self):
		return ('PotParam: %f %f '%(self.rmin,self.rmax)) + str(self.coeff)
	def __repr__(self):
		return ('PotParam: %f %f '%(self.rmin,self.rmax)) + str(self.coeff)

	def SetRepulsive(self, rmin, rmax, A, alpha):
		self.rmin = rmin
		self.rmax = rmax
		self.coeff = dict(A=A,alpha=alpha)
		return self


	def SetMorseX(self, rmin, rmax, D0, alpha, r0, ron):
		self.rmin = rmin
		self.rmax = rmax
		self.coeff = dict(D0=D0,alpha=alpha,r0=r0,ron=ron)
		return self




"""
parameters:
	- potentials -- list of potential objects corresponding to the potentials being used
	- matrix -- matrix of size n_t by n_t (where n_t is the number of particle types). Each element is a list of PotParam objects
"""
class Interactions:
	def __init__(self,potentials=[],n_t=0):
		self.potentials = potentials
		self.InitializeMatrix(n_t)

	def InitializeMatrix(self, n_t):
		npotentials = len(self.potentials)
		self.matrix = [ [ [None for i in range(npotentials)] for j in range(n_t)] for k in range(n_t)]

	def CheckSymmetric(self):
		assert(True)


#interaction.potentials
#interaction.matrix
	#interaction.matrix[i,j]=[None,params]
#params is a structure that contains rmin, rmax and coeff







"""
SystemDefinition class
This class completely defines the setup of a system (what the building blocks are, how they interact, etc.)
parameters:
	buildingBlockTypeList -- list of BuildingBlockType objects (size=m, where m is the number of building block types)
	particleTypes -- list of ParticleType objects (size=n_t, where n_t is the number of particle types)
	interactions -- something that contains all the interaction data between all n_t x n_t ParticleType pairs... not sure yet what this will look like.
	buildingBlockNumbers -- list of how many of each buildingBlockType (size=m)
	L -- linear length of system: volume = L^DIM
	kBT -- temperature in units of kB
	seed -- seed for the random number generator
	basename -- a string to be used for all input and output. Should it include a directory path? The idea is that, for example, the trajectory will be traj_fn = basename+"_trajectory.gsd", etc. etc.
"""
#note: we can probably integrate the Params class into this... maybe it will contain a params object



class SystemDefinition:
        def __init__(self):
                self.buildingBlockTypeList = []
                self.particleTypes = []
                self.interactions = Interactions()
                #self.buildingBlockNumbers = []
                self.L = 0.
                self.kBT = 0.
                self.seed = 12345
                self.basename = "test"

        def GetTypeStrings(self):
                return np.array([ pt.typeString for pt in self.particleTypes ])

        #for a given building block, get the list of strings for the type of each particle
        def GetParticleTypeStrings(self, bbi):
                return np.array([ self.particleTypes[tid].typeString for tid in self.buildingBlockTypeList[bbi].typeids])


        # To use this function:

        # SystDef_fn = open('%sSystDef.txt' % args.input_name,'w')
        # SystDef.saveToFile(SystDef_fn)
        # listFilename.close()

        def saveToFile(self, outfn):

                # line 0 = self.particleTypes
                PartTypesNumber = len(self.particleTypes)
                outfn.write("{} ".format(PartTypesNumber))
                for i in range(PartTypesNumber):
                        outfn.write("{} {} ".format(self.particleTypes[i].typeString,self.particleTypes[i].radius))
                outfn.write("\n")

                # line 1 = BBTypesNumber self.buildingBlockTypeList[0].n self.buildingBlockTypeList[1].n self.buildingBlockTypeList[2].n etc
                BBTypesNumber = len(self.buildingBlockTypeList)
                outfn.write("{} ".format(BBTypesNumber))
                for i in range(BBTypesNumber):
                        outfn.write("{} ".format(self.buildingBlockTypeList[i].n))
                outfn.write("\n")

                # line 2 to self.buildingBlockTypeList
                for i in range(BBTypesNumber):
                        #outfn.write("{} ".format(self.buildingBlockTypeList[i].n))
                        for pos in self.buildingBlockTypeList[i].positions:
                                outfn.write("{} {} {} ".format(pos[0],pos[1],pos[2]))
                        outfn.write("\n")

                PotentialsNumber = len(self.interactions.potentials)
                outfn.write("{} ".format(PotentialsNumber))
                for i in range(PotentialsNumber):
                        outfn.write("{} ".format(self.interactions.potentials[i].__class__.__name__))
                outfn.write("\n")

                for i in range(PartTypesNumber):
                        for j in range(PartTypesNumber):
                                outfn.write("{},".format(self.interactions.matrix[i][j]))
                        outfn.write("\n")
                outfn.write("\n")

        ############################################################################################


        def loadFromFile(self, infn):

                lines = infn.readlines()

                # line 0 = particle types
                PartTypesNumber = int(lines[0].split(' ')[0])
                self.particleTypes = [ ParticleType() for i in range(PartTypesNumber) ]
                for i in range(PartTypesNumber):
                        self.particleTypes[i].typeString = str(lines[0].split(' ')[(i*2)+1])
                        self.particleTypes[i].radius = str(lines[0].split(' ')[(i*2)+2])

                # line 1 = number of bbtypes
                BBTypesNumber = int(lines[1].split(' ')[0])

                # line 2 = building blocks
                self.buildingBlockTypeList = [ BuildingBlockType() for i in range(BBTypesNumber) ]

                for i in range(BBTypesNumber): # only 1 for now
                        self.buildingBlockTypeList[i].n = int(lines[1].split(' ')[i+1])
                        self.buildingBlockTypeList[i].positions = np.zeros((self.buildingBlockTypeList[i].n,3),dtype=float)

                        for j in range(self.buildingBlockTypeList[i].n):
                                self.buildingBlockTypeList[i].positions[j,0] = lines[2].split(' ')[(j*3)]
                                self.buildingBlockTypeList[i].positions[j,1] = lines[2].split(' ')[(j*3)+1]
                                self.buildingBlockTypeList[i].positions[j,2] = lines[2].split(' ')[(j*3)+2]

                PotentialsNumber = int(lines[3].split(' ')[0])
                Pots = []
                module = __import__('potentials')
                for i in range(PotentialsNumber):
                        PotName = lines[3].split(' ')[i+1]
                        PotClass = getattr(potentials, PotName)
                        Pots.append(PotClass())
                self.interactions.potentials = Pots

                # for i in range(PartTypesNumber):
                #         for j in range(PartTypesNumber):






        #for each particle in the building block
        #rotate by q and then translate by t
        def TransAndRotBB(self, t, q, bbT):
                Rrot = transformations.quaternion_matrix(q)
                Rtrans = transformations.translation_matrix(t)
                T = np.matmul(Rtrans,Rrot)
                return np.array([np.matmul(T,np.concatenate((p,[1])))[:3] for p in self.buildingBlockTypeList[bbT].positions])


        # translate by t and rotate by p particle at position p
        # p is the position of the particle relative to the COM of the builfding block it belongs to
        def TransAndRotPart(self, t, q, p):
                Rrot = transformations.quaternion_matrix(q)
                Rtrans = transformations.translation_matrix(t)
                T = np.matmul(Rtrans,Rrot)
                return np.array(np.matmul(T,np.concatenate((p,[1])))[:3])


        def SaveStateToXYZ_Full(self,outfn, state, command='w', comment=''):
                with open(outfn,command) as outfile:
                        #get total number of particles (not including com particles
                        NPart = np.sum(np.array([self.buildingBlockTypeList[bbt].n for bbt in state.bbTypes]))
                        #NPart = np.sum(np.array([BBT.n for BBT in self.buildingBlockTypeList]))

                        outfile.write('%d\n' % NPart)
                        outfile.write('#' + comment + '\n')
                        for pos,ori,bbT in zip(state.positions,state.orientations,state.bbTypes):
                                rpos = self.TransAndRotBB(pos,ori,bbT)
                                for p,tid in zip(rpos,self.buildingBlockTypeList[bbT].typeids):
                                        typeString = self.particleTypes[tid].typeString
                                        outfile.write('{} {} {} {}\n'.format(typeString,p[0],p[1],p[2]))


        def SaveStateToXYZ_COMParticles(self,outfn, state, command='w', comment=''):
                with open(outfn,command) as outfile:
                        NBB = len(state.positions)
                        outfile.write('%d\n' % NBB)
                        outfile.write('#' + comment + '\n')
                        for pos,ori,bbT in zip(state.positions,state.orientations,state.bbTypes):
                                outfile.write('{} {} {} {} {} {} {} {}\n'.format(bbT, pos[0],pos[1],pos[2],ori[0],ori[1],ori[2],ori[3]))



        #Right now, only read the first state in the file...
	#TODO: allow for reading state n
        def LoadStateFromXYZ_COMParticles(self, infn):
                state = SystemState()
                import csv
                with open(infn,'r') as infile:
                        reader = csv.reader(infile,delimiter=' ')

                        #read the number of building blocks
                        row = reader.__next__()
                        nBB = int(row[0])

                        #throw away the comment line
                        reader.__next__()

                        #loop over the rest of the rows
                        bbTypes = []
                        poss = []
                        oris = []
                        for row in reader:
                                bbTypes.append(int(row[0]))
                                poss.append(np.float64(row[1:4]))
                                oris.append(np.float64(row[4:]))
                        state.positions = np.array(poss)
                        state.orientations = np.array(oris)
                        state.bbTypes = np.array(bbTypes)
                return state




"""
SystemState class
This class completely defines the current state of a system given some SystemDefinition object
parameters:
	positions -- matrix with shape (N,DIM) where N is the number of building blocks. Position of the center of mass of every building block
	orientations -- matrix with shape (N,4). Quaternion that gives the orientation of every building block.
	bbTypes -- array of length N. The index of the bulding block type. If the state is associated with the SystemDefinition sysDef, then the type of building block i is sysDef.buildingBlockTypeList[self.bbTypes[i]]
"""

class SystemState:
	def __init__(self):
		self.positions = np.array([])
		self.orientations = np.array([])
		self.bbTypes = np.array([])

	def Save(self,fn):
		pass

	def Load(self,fn):
		pass
