from __future__ import division
import random
import math
import numpy as np
from classDefinitions import *
from potentials import *
import argparse

import hoomd
import hoomd.md
import gsd.hoomd


# get position of a particle
def GetInitPositionTheta(i, j, k, n2, n3, theta, a1):
	Rk = n2/theta - (n3-k-1)
	thetaj = (j+0.5)*theta/(n2-1)
	x = i*a1
	y = Rk*math.sin(thetaj)
	z = Rk*math.cos(thetaj)
	return [x,y,z]



def GetType(i, j, k, n2, typePrefix1, typePrefix2):
	if j==0:
		#return typePrefix1 + str(k) + str(i)
		return typePrefix1 + '{:03d}'.format(k) + '_{:03d}'.format(i)
		#return "A"
	if j==n2-1:
		#return typePrefix2 + str(k)
		return typePrefix2 + '{:03d}'.format(k) + '_{:03d}'.format(i)
		#return "B"
	return 'D' + '{:01d}'.format(k)

#assume every particle has unit mass
def CalculateCOM(positions):
	com = np.array([0.,0.,0.])
	for p in positions:
		com += p
	com /= len(positions)
	return com

def CalculateMomentOfInertiaDiag(e, positions):
        #Isphere = (2./5.)*(0.5**2)  #should we include this?????
        I = 0.
        for p in positions:
                #d = math.sqrt( np.dot(p,p) - (np.dot(p,e)**2)/np.dot(e,e) )
                d = np.dot(p,p) - (np.dot(p,e)**2)/np.dot(e,e)
                I += d
        return I

def CalculateMomentOfInertiaOffDiag(e1, e2, positions):
        #	Isphere = (2./5.)*(0.5**2)  #should we include this?????
        I = 0.
        for p in positions:
                d = np.dot(p,e1) * np.dot(p,e2)
                I -= d
        return I

def CalculateMomentsOfInertia(positions):
        Ix = CalculateMomentOfInertiaDiag(np.array([1,0,0]),positions)
        Iy = CalculateMomentOfInertiaDiag(np.array([0,1,0]),positions)
        Iz = CalculateMomentOfInertiaDiag(np.array([0,0,1]),positions)
        return [Ix,Iy,Iz]

def CalculatePrincipalMomentsOfInertia(positions):
        Ixx = CalculateMomentOfInertiaDiag(np.array([1,0,0]),positions)
        Iyy = CalculateMomentOfInertiaDiag(np.array([0,1,0]),positions)
        Izz = CalculateMomentOfInertiaDiag(np.array([0,0,1]),positions)

        Ixy = CalculateMomentOfInertiaOffDiag(np.array([1,0,0]), np.array([0,1,0]), positions)
        Iyz = CalculateMomentOfInertiaOffDiag(np.array([0,1,0]), np.array([0,0,1]), positions)
        Izx = CalculateMomentOfInertiaOffDiag(np.array([0,0,1]), np.array([1,0,0]), positions)

        I = np.array([[Ixx,Ixy,Izx],[Ixy,Iyy,Iyz],[Izx,Iyz,Izz]])

        evals, evecs = np.linalg.eigh(I)

        return evals, evecs

def GetSphereRadiiList(n2,n3,theta):
	rs = [ (n2/theta - (n3-k-1)) * theta/(2.*(n2-1)) for k in range(n3) ]
	#print(rs)
	return rs

def GetRadii(k,n2,n3,theta):
	rs = (n2/theta - (n3-k-1)) * theta/(2.*(n2-1))
	return rs

# input:

# n1 - "height": number of spheres that make each helix

# n2 - "length": number of helices in the "long" axis. Must be greater
#                than or equal to 2 (this should be explicitly checked)

# n3 - "width": number of helices in the "short" axis. Must be greater
#               than or equal to 2, but will probably often be exactly 2.

# theta - determines the bend in the lattice. The lattice will be
#         mapped to a cylinder, and this is the angle over which it
#         bends. Must be less than or equal to 2*pi.

# a1 - lattice spacing in the "height" directions. Decreasing this
#      from 1 while keeping a1*n1 fixed lead to a better approximation of a
#      cylinder for each helix

# typePrefix1 - the types for the first side will be typePrefix1
#               followed by a number

# typePrefix2 - the types for the second side will be typePrefix2
#               followed by a number

def InitBuildingBlockTheta(n1, n2, n3, theta, a1, typePrefix1, typePrefix2):
        positionsTemp = []
        typesTemp = []
        radiiTemp = []
        for i in range(0,n1):
                for j in range(0,n2):
                        for k in range(0,n3):
                                pos = GetInitPositionTheta(i,j,k,n2,n3,theta,a1)
                                positionsTemp.append(pos)

                                typ = GetType(i,j,k,n2,typePrefix1,typePrefix2)
                                typesTemp.append(typ)

                                rad = GetRadii(k,n2,n3,theta)
                                radiiTemp.append(rad)

        positions = np.array(positionsTemp)
        types = typesTemp
        radii = radiiTemp

        mass = len(positions)

	#Calculate the center of mass
        com = CalculateCOM(positions)

	#Subtract off the center of mass
        for i in range(len(positions)):
                positions[i] -= com

        evals_I, evecs_I = CalculatePrincipalMomentsOfInertia(positions)

        #print("I evals evecs: \n", evals_I, evecs_I)

        R = evecs_I.T

        for i in range(len(positions)):
                positions[i] = np.dot(R,positions[i])

	#return everything
        return (positions,types,mass,evals_I,radii)




def InitializeSystem_overlap(params):

        (p,t,m,I,r) = InitBuildingBlockTheta(params.n1,params.n2,params.n3,params.theta,params.a1,params.typePrefix1,params.typePrefix2)

        # List of unique types, the corresponding indices, and the
        t_unique,i_unique,typeids = np.unique(t,return_index=True,return_inverse=True)

        # Empty list of radii
        r_unique = []

        # List of radii corresponding to the unique types
        for i in i_unique:
                r_unique.append(r[i])

        # Number of unique types
        n_t = len(t_unique)

        # Create and populate the particle types with type name and radius
        PartTypes = [ ParticleType() for i in range(n_t) ]
        for i in range(n_t):
                PartTypes[i].typeString = t_unique[i]
                PartTypes[i].radius = r_unique[i]

        # Create and populate the building block type: in this case only one
        BBlockTypeList = []

        BBlockType = BuildingBlockType() # for i in range(n_bb)
        BBlockType.n = len(p)
        BBlockType.positions = np.array(p)
        BBlockType.typeids = typeids
        BBlockType.mass = m
        BBlockType.moment_inertia = I

        BBlockTypeList.append(BBlockType)

        # List of potentials
        Pots = [ RepulsivePotential(), MorseXPotential()]

        # Create the interaction object
        Inter = Interactions()
        Inter.potentials = Pots
        Inter.InitializeMatrix(n_t)

        morse_rcut = 8./params.morse_a + params.morse_r0

        # Populate interaction matrix
        for i in range(n_t):
                for j in range(n_t):
                        if( t_unique[i][0] == t_unique[j][0] ):
                                # Repulsion
                                Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)
                        elif( t_unique[i][0] != t_unique[j][0] and t_unique[i][1:] == t_unique[j][1:] ):
                                # Attraction
                                Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )
                        #print(PartTypes[i].typeString, PartTypes[j].typeString, Inter.matrix[i][j])


        # Create and populate the system definition object
        SystDef = SystemDefinition()
        SystDef.buildingBlockTypeList = BBlockTypeList
        SystDef.particleTypes = PartTypes
        SystDef.interactions = Inter
        SystDef.concentration = params.concentration
        SystDef.Lxyz = 10.
        SystDef.kBT = params.kT_brown
        SystDef.seed = random.randint(1,1001)
        SystDef.basename = params.fnbase

        return SystDef





def InitBuildingBlockDimers(Type):

        positions = [[0.,0.,0.],[1.,0.,0.]]
        if Type == 0:
                types = ["A","X"]
        elif Type ==1:
                types = ["A","Y"]
        radii = [1.,0.25]
        mass = 1
        # moment of inertia of a sphere
        I = 2./5.

	#return everything
        return (positions,types,mass,[I,I,I],radii)





def InitializeSystem_dimers(params):

        p = []
        t = []
        m = []
        I = []
        r = []

        for i in range(2):
                (p_temp,t_temp,m_temp,I_temp,r_temp) = InitBuildingBlockDimers(i)
                p.append(p_temp)
                t.append(t_temp)
                m.append(m_temp)
                I.append(I_temp)
                r.append(r_temp)


        # List of unique types, the corresponding indices, and the
        t_unique,i_unique,typeids = np.unique(t,return_index=True,return_inverse=True)

        # Empty list of radii
        rsum = []
        for i in range(len(r)):
                rsum+=r[i]

        r_unique = [ rsum[i] for i in i_unique ]

        # Number of unique types
        n_t = len(t_unique)

        # Create and populate the particle types with type name and radius
        PartTypes = [ ParticleType() for i in range(n_t) ]
        for i in range(n_t):
                PartTypes[i].typeString = t_unique[i]
                PartTypes[i].radius = r_unique[i]# Create and populate the particle types with type name and radius

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

                count+=BBlockType.n

        # List of potentials
        Pots = [ RepulsivePotential(), MorseXPotential(), MorseXRepulsivePotential()]

        # Create the interaction object
        Inter = Interactions()
        Inter.potentials = Pots
        Inter.InitializeMatrix(n_t)

        morse_rcut = 8./params.morse_a + params.morse_r0

        # Populate interaction matrix
        for i in range(n_t):
                for j in range(n_t):
                        if( t_unique[i] == t_unique[j] and t_unique[i] == 'A'):
                                # Repulsion between real spheres
                                Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)
                        elif( t_unique[i] == t_unique[j] and t_unique[i] != 'A'):
                                # Repulsion between attractors of the same color
                                Inter.matrix[i][j][2] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )
                        elif( t_unique[i] != t_unique[j] and t_unique[i] != 'A' and t_unique[j] != 'A' ):
                                # Attraction
                                Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )


        # Create and populate the system definition object
        SystDef = SystemDefinition()
        SystDef.buildingBlockTypeList = BBlockTypeList
        SystDef.particleTypes = PartTypes
        SystDef.interactions = Inter
        SystDef.concentration = params.concentration
        SystDef.Lxyz = 10.
        SystDef.kBT = params.kT_brown
        SystDef.seed = random.randint(1,1001)
        SystDef.basename = params.fnbase

        return SystDef





def InitBuildingBlockDimers_tr(Type):

        a = 1 # distance of the center of the spheres from the BB COM
        b = .3 # distance of the center of the patches from the BB COM

        positions = [ [0.,                  0.,                   a], # first sphere
                      [0.,  a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)], # second sphere
                      [0., -a*np.cos(np.pi/6.), -a*np.sin(np.pi/6.)], # third sphere
                      [a,                   0.,                   b], # first patch
                      [a,   b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)], # second patch
                      [a,  -b*np.cos(np.pi/6.), -b*np.sin(np.pi/6.)]  # third patch
              ]

        if Type == 0:
                types = ["A", "A", "A", "B1", "R1", "G1"]
        elif Type ==1:
                types = ["A", "A", "A", "B2", "G2", "R2"]
        radii = [a, a, a, 0.2*a, 0.2*a, 0.2*a]
        mass = 1

        # principal moments of inertia of the principal trimer - the patches don't count
        #evals_I, evecs_I = CalculatePrincipalMomentsOfInertia(positions[0:3])

        # calculated with mathematica as point particles
        #evals_I = np.array([3.,1.5,1.5])

        """used on simulations of 7-21-2020"""
        #evals_I = np.array([1.,1.,1.])

        """used on simulations of 8-4-2020"""
        evals_I = np.array([3.,1.5,1.5])


	#return everything
        return (positions,types,mass,evals_I,radii)





def InitializeSystem_dimers_tr(params):

        p = []
        t = []
        m = []
        I = []
        r = []

        for i in range(2):
                (p_temp,t_temp,m_temp,I_temp,r_temp) = InitBuildingBlockDimers_tr(i)
                p.append(p_temp)
                t.append(t_temp)
                m.append(m_temp)
                I.append(I_temp)
                r.append(r_temp)


        # List of unique types, the corresponding indices, and the
        t_unique,i_unique,typeids = np.unique(t,return_index=True,return_inverse=True)

        # Empty list of radii
        rsum = []
        for i in range(len(r)):
                rsum+=r[i]

        r_unique = [ rsum[i] for i in i_unique ]

        # Number of unique types
        n_t = len(t_unique)

        # Create and populate the particle types with type name and radius
        PartTypes = [ ParticleType() for i in range(n_t) ]
        for i in range(n_t):
                PartTypes[i].typeString = t_unique[i]
                PartTypes[i].radius = r_unique[i]# Create and populate the particle types with type name and radius

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

                count+=BBlockType.n

        # List of potentials
        Pots = [ RepulsivePotential(), MorseXPotential(), MorseXRepulsivePotential()]

        # Create the interaction object
        Inter = Interactions()
        Inter.potentials = Pots
        Inter.InitializeMatrix(n_t)

        morse_rcut = 8./params.morse_a + params.morse_r0

        # # Populate interaction matrix
        # for i in range(n_t):
        #         for j in range(n_t):
        #                 if( t_unique[i] == t_unique[j] and t_unique[i] == 'A'):
        #                         # Repulsion between real spheres
        #                         Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)
        #                 elif( t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1]):
        #                         # Attraction between patches of the same color (e.g. G1 and G2)
        #                         Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )
        #                 elif( t_unique[i] == t_unique[j][0] and t_unique[i] != 'A'):
        #                         # Repulsion between patches of the same type (e.g. two G1's)
        #                         Inter.matrix[i][j][2] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )

        """
        Model updated on July 29 2020 to consider different D0 for the different patches
        morse_D0 = 7.0
        morse_D0_r = 1.0 # 1.0
        morse_D0_g = 1.5 # 0.5
        morse_D0_b = 2.0 # 3.0

        """

        """
        Model updated on August 11 2020 to fix a bug with same patches attracting

        """

        # Populate interaction matrix
        for i in range(n_t):
                for j in range(n_t):
                        if( t_unique[i] == t_unique[j] and t_unique[i] == 'A' ):
                                # Repulsion between real spheres
                                Inter.matrix[i][j][0] = PotParam().SetRepulsive(0, PartTypes[i].radius+PartTypes[j].radius, params.rep_A, params.rep_alpha)
                        elif( t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'R' ):
                                # Attraction between red patches
                                Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0*params.morse_D0_r, params.morse_a, params.morse_r0, morse_rcut/2. )
                        elif( t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'G' ):
                                # Attraction between green patches
                                Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0*params.morse_D0_g, params.morse_a, params.morse_r0, morse_rcut/2. )
                        elif( t_unique[i][0] == t_unique[j][0] and t_unique[i][1] != t_unique[j][1] and t_unique[i][0] == 'B' ):
                                # Attraction between blue patches
                                Inter.matrix[i][j][1] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0*params.morse_D0_b, params.morse_a, params.morse_r0, morse_rcut/2. )
                        elif( t_unique[i] == t_unique[j] and t_unique[i] != 'A' ):
                                # Repulsion between patches of the same type (e.g. two G1's)
                                Inter.matrix[i][j][2] = PotParam().SetMorseX( 0, morse_rcut, params.morse_D0, params.morse_a, params.morse_r0, morse_rcut/2. )


        # Create and populate the system definition object
        SystDef = SystemDefinition()
        SystDef.buildingBlockTypeList = BBlockTypeList
        SystDef.particleTypes = PartTypes
        SystDef.interactions = Inter
        SystDef.concentration = params.concentration
        SystDef.Lxyz = 10.
        SystDef.kBT = params.kT_brown
        SystDef.seed = random.randint(1,1001)
        SystDef.basename = params.fnbase

        return SystDef





def SaveParamFile(params, filename):
        #attributes = [a for a in dir(params) if not a.startswith('__') and not callable(getattr(params,a))]
        attributes = [a for a in params.__dir__() if not a.startswith('__') and not callable(getattr(params,a))]
        values = [getattr(params,a) for a in attributes]
        assert len(attributes)==len(values)
        with open(filename,"w") as text_file:
                for (a,v) in zip(attributes,values):
                        text_file.write("%s %s\n" % (a, repr(v)))




def parserFunc():
        parser = argparse.ArgumentParser(description='Run a simulation.')
        parser.add_argument('-i', '--input_name', type=str,            default='test', help='input file')
        parser.add_argument('-D', '--morse_D0',   type=float,          default=7.,     help='morse min')
        parser.add_argument('-a', '--morse_a',    type=float,          default=5.,     help='morse strength')
        parser.add_argument('-c', '--conc',       type=float,          default=0.001,  help='concentration')
        parser.add_argument('-s', '--string',     type=str,            default='test', help='output file name')
        parser.add_argument('-t', '--testing',    action="store_true", default=False,  help='put the simulation in testing mode')
        parser.add_argument('-f', '--save_file',  action="store_true", default=False,  help='save snaps of the clusters found')

        return parser





def RunSimulation(params,SystDef):

        SaveParamFile(params,SystDef.basename+"_params.txt")

        PartTypes  = SystDef.particleTypes
        BBlockTypeList = SystDef.buildingBlockTypeList

        NbbTypes = len(BBlockTypeList)
        C = ["C0","C1"]

        # Create a unit cell that contains one structure (protein)
        uc = hoomd.lattice.unitcell(
                N = NbbTypes,                               # one particle per unit cell
                a1 = [params.ls,         0,         0],     # lattice spacing in the x direction
                a2 = [ 0,        params.ls,         0],     # lattice spacing in the y direction
                a3 = [ 0,                0, params.ls],     # lattice spacing in the z direction
                dimensions = 3,
                position = [[0,0,0],[params.ls/2.,params.ls/2.,params.ls/2.]],  # the position of this central particle is the center of mass of the protein
                type_name = C,     # the center of mass is a dumb particle
                mass = [ BBlockTypeList[i].mass for i in range(NbbTypes) ],            # whose mass is the mass of the protein
                moment_inertia = [ BBlockTypeList[i].moment_inertia for i in range(NbbTypes) ],
                orientation = [[1, 0, 0, 0],[1, 0, 0, 0]]
        );

        Ntot = 0
        for i in range(NbbTypes):
                Ntot += params.N[i]

        # Create a lattice of ixjxk unit cells (particles)
        nx = ny = nz = math.ceil( params.N[0] ** (1./3.) )

        system = hoomd.init.create_lattice(unitcell=uc, n=[nx,ny,nz]);

        # Pick some particles at random
        Ntemp = nx*ny*nz*2
        print(Ntemp,len(system.particles))
        assert Ntemp == len(system.particles)
        assert Ntemp >= Ntot
        Nremove = Ntemp - Ntot
        print("Nremove = ", Nremove)
        if(Nremove > 0):
                tagsToRemove = []
                for pi in random.sample(range(Ntemp),Nremove):
                        tagsToRemove.append(system.particles[pi].tag)

                for tag in tagsToRemove:
                        print("Removing particle with tag = ", tag)
                        system.particles.remove(tag)

        # Only unique types are listed
        unique_types = SystDef.GetTypeStrings()

        for ut in unique_types:
                system.particles.types.add(ut)


        # Creates an object rigid body
        rigid = hoomd.md.constrain.rigid()

        # Populate the rigid body object
        for bbN in range(NbbTypes):
                rigid.set_param(C[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

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

        # Saves time, energy in a text file -- not really needed
        #hoomd.analyze.log(filename='%s_Data' % params.fnbase, quantities=['time','potential_energy'], period=params.Trec_data, header_prefix='#', overwrite=True, phase=0)

        # Concentration c=N/L^3 is the input to get the final (after shrinking) box size
        Linit  = params.ls*nx;
        Lfinal = (Ntot/params.concentration) ** (1./3.)
        print("Linit  = ", Linit,  " Vinit  = ", Linit**3.0)
        print("Lfinal = ", Lfinal, " Vfinal = ", Lfinal**3.0)
        hoomd.update.box_resize(L = hoomd.variant.linear_interp([(0,Linit), (1e5, Lfinal)]))

        # Dump trajectories in a gsd file
        hoomd.dump.gsd("%s_Traj.gsd" % params.fnbase,
                period=params.Trec_traj,
                #group=rigid, FOR LATER
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



def ReRunSimulation(params,SystDef,infbase):

        SaveParamFile(params,SystDef.basename+"_params.txt")

        basename=infbase
        inputname=basename+'_Traj.gsd'

        full_traj = gsd.hoomd.open(inputname,'rb')
        len_traj = len(full_traj)
        print(len_traj)

        input_snap = hoomd.data.gsd_snapshot(inputname, frame=len_traj-1)

        PartTypes  = SystDef.particleTypes
        BBlockTypeList = SystDef.buildingBlockTypeList

        NbbTypes = len(BBlockTypeList)
        C = ["C0","C1"]

        system = hoomd.init.read_snapshot(input_snap)

        # Only unique types are listed
        unique_types = SystDef.GetTypeStrings()

        for ut in unique_types:
                system.particles.types.add(ut)


        # Creates an object rigid body
        rigid = hoomd.md.constrain.rigid()

        # Populate the rigid body object
        for bbN in range(NbbTypes):
                rigid.set_param(C[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

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

        # Saves time, energy in a text file -- not really needed
        #hoomd.analyze.log(filename='%s_Data' % params.fnbase, quantities=['time','potential_energy'], period=params.Trec_data, header_prefix='#', overwrite=True, phase=0)

        # Concentration c=N/L^3 is the input to get the final (after shrinking) box size
        # Linit  = params.ls*nx;
        # Lfinal = (Ntot/params.concentration) ** (1./3.)
        # print("Linit  = ", Linit,  " Vinit  = ", Linit**3.0)
        # print("Lfinal = ", Lfinal, " Vfinal = ", Lfinal**3.0)
        # hoomd.update.box_resize(L = hoomd.variant.linear_interp([(0,Linit), (1e5, Lfinal)]))

        # Dump trajectories in a gsd file
        hoomd.dump.gsd("%s_Traj.gsd" % params.fnbase,
                period=params.Trec_traj,
                #group=rigid, FOR LATER
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
