from __future__ import division
import random
import sys
import math
import numpy as np
from RunFunctions import *
from dynamicalMatrix_4 import *
import argparse

import hoomd
import hoomd.md
import gsd.hoomd

import rowan
from scipy.optimize import minimize


# Initialize the simulation
#hoomd.context.initialize("");




def SaveParamFile(params, filename):
        #attributes = [a for a in dir(params) if not a.startswith('__') and not callable(getattr(params,a))]
        attributes = [a for a in params.__dir__() if not a.startswith('__') and not callable(getattr(params,a))]
        values = [getattr(params,a) for a in attributes]
        assert len(attributes)==len(values)
        with open(filename,"w") as text_file:
                for (a,v) in zip(attributes,values):
                        text_file.write("%s %s\n" % (a, repr(v)))


def GDEnMin(systDef, state):

        eps = 1e-5
        delta_E = 1
        delta_G = 1
        gamma = 1e-6

        BBlocks_num = len(state.bbTypes)

        G, Etot = mainGradientCalc(systDef, state)

        G_norm_old = np.linalg.norm(G)
        G_norm_new = np.linalg.norm(G)

        count = 0

        while delta_G > eps:

                euler = np.array([ rowan.to_euler(q, convention='zyz') for q in np.array(rowan.normalize(state.orientations)) ])

                if not (count%1e2):
                        print("delta_E", delta_E, "delta_G", delta_G)
                        #print("euler",euler)


                for bb in range(BBlocks_num):
                        for j in range(3):
                                # state.positions[bb][j] -= gamma * G[bb*6+j]
                                # euler[bb][j] -= gamma * G[bb*6+3+j]

                                state.positions[bb][j] = state.positions[bb][j]*(1-gamma) - gamma * G[bb*6+j]
                                euler[bb][j] = euler[bb][j]*(1-gamma) - gamma * G[bb*6+3+j]

                state.orientations = np.array([ rowan.from_euler(eu[0],eu[1],eu[2], convention='zyz') for eu in euler ])

                G, Etot = mainGradientCalc(systDef, state)

                G_norm_new = np.linalg.norm(G)
                G_norm_old = G_norm_new

                delta_G = abs(G_norm_old - G_norm_new)
                delta_E = abs(Etot + 50)

                count += 1

        print("\ncount",count)
        print("force", G)
        print("energy", Etot)
        print("positions",state.positions)
        print("orientations",state.orientations)





def ScipyEnMin(systDef, state): # with scipy.optimize

        eps = 1e-3
        delta = 1
        gamma = 1e-4

        BBlocks_num = len(state.bbTypes)

        bbPos = state.positions
        bbOr = state.orientations

        x0 = np.concatenate((bbPos,bbOr),axis=1)

        bbTyp = state.bbTypes

        def fun(x,bbt,system):

                M = len(bbt)

                state_f = SystemState()

                pos_f = []
                ori_f = []

                for i in range(M):
                        pos_f.append(x[i*7 : i*7+3])
                        ori_f.append(x[i*7+3 : i*7+7])

                state_f.positions = np.array(pos_f)
                state_f.orientations = np.array(ori_f)
                state_f.bbTypes = np.array(bbt)

                G, Etot = mainGradientCalc(system, state_f)
                return Etot


        #res = minimize(fun, x0, args=(bbTyp,systDef), method='CG', tol=1e-4, options={'disp': True})
        res = minimize(fun, x0, args=(bbTyp,systDef), method='CG', options={'gtol': 1e-11,'disp': True})

        M = len(bbTyp)

        pos = []
        ori = []

        for i in range(M):
                pos.append(res.x[i*7 : i*7+3])
                ori.append(res.x[i*7+3 : i*7+7])

        state.positions = np.array(pos)
        state.orientations = np.array(ori)




def FireTotEnMin(params,SystDef,curr_snap,inputname):

        SaveParamFile(params, params.fnbase+"_params.txt")

        PartTypes  = SystDef.particleTypes
        BBlockTypeList = SystDef.buildingBlockTypeList

        snap = hoomd.data.gsd_snapshot(inputname, frame=curr_snap)

        types_lett = snap.particles.types # Names of particle types

        system = hoomd.init.read_snapshot(snap)

        # Only unique types are listed

        unique_types = SystDef.GetTypeStrings()

        for ut in unique_types:
                system.particles.types.add(ut)

        C = ["C0","C1"]

        # Creates a list of objects rigid body
        rigid = hoomd.md.constrain.rigid()

        for bbN in range(len(BBlockTypeList)):

                # Populate the rigid body: to the dumb particle attach all the others of types and positions
                rigid.set_param(C[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

        # Creates the rigid body object
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

        # Groups particles
        rigid = hoomd.group.rigid_center();

        # Number of rigid bodies
        N = len(rigid)
        #assert N==params.N

        # Applies integration scheme to the group
        # the one found with Carl:
        # fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=1, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-11, wtol=1e-11, Etol=1e-11, min_steps=500, group=rigid, aniso=True)

        fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=1, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-11, wtol=1e-11, Etol=1e-11, min_steps=500, group=rigid, aniso=True)

        #fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-7, wtol=1e-7, Etol=1e-07, min_steps=10, group=rigid, aniso=True)

        #hoomd.md.integrate.mode_standard(dt=params.dT);

        #nve = hoomd.md.integrate.nve(group=rigid)
        #hoomd.md.integrate.brownian(group=rigid, kT=0.0, seed=params.seed_brown)

        # Save total potential energy
        hoomd.analyze.log(filename='%s_Data' % params.fnbase, quantities=['time','potential_energy'], period=params.Trec_data, header_prefix='#', overwrite=True, phase=0)

        # Saves trajectories in a gsd file
        hoomd.dump.gsd("%s_EMin.gsd" % params.fnbase,
                       period=params.Trec_traj,
                       group=hoomd.group.all(),
                       #group=rigid,
                       overwrite=True,
                       dynamic =['attribute']
        );


        # callback function to save the total net force of the system on a file, period equal to the saving period of the energy
        ForceFile = open('%sForce' % params.fnbase,'w')

        def force(time_step):
                f = 0
                for p in system.particles:
                        f += np.linalg.norm(p.net_force)
                #TF = fire.has_converged()
                #ForceFile.write("{} {} {} {} {}\n".format(time_step,f[0],f[1],f[2],TF))
                ForceFile.write("{} {}\n".format(time_step,f))
                ForceFile.flush()
                #print(fire.has_converged())

        # Run like hell
        # f_norm = 1

        # while(f_norm > 1e-3):
        #         print("Total force:", f_norm)
        #         hoomd.run(params.Ttot, callback_period=100000, callback=force)
        #         f_norm = 0
        #         for p in system.particles:
        #                 f_norm += np.linalg.norm(p.net_force)



        counter = 0

        while not(fire.has_converged()):
                if counter>1000:
                        break
                hoomd.run(params.Ttot, callback_period=params.Trec_data, callback=force)
                counter += 1

        #hoomd.run(1000, callback_period=params.Trec_data, callback=force)

        if not fire.has_converged():
                print("\n")
                print("WARNING: convergence not reached!")
                print("\n")
        else:
                # I am happy
                print("\n")
                print("so far so good")
                print("\n")


        ForceFile.close()





def FireSingleEnMin(params,SystDef,state,str_j):

        PartTypes  = SystDef.particleTypes
        BBlockTypeList = SystDef.buildingBlockTypeList

        N_state = len(state.bbTypes)

        COM_names = []

        for tn in state.bbTypes:
                COM_names.append("C%s" % tn)

        hoomd.context.initialize("");

        snapshot = hoomd.data.make_snapshot(N=N_state,
                                            box=hoomd.data.boxdim(L=10),
                                            particle_types=COM_names,
                                    )

        all_names = []

        for n,bbN in enumerate(state.bbTypes):
                snapshot.particles.position[n]       = state.positions[n]
                snapshot.particles.orientation[n]    = state.orientations[n]
                snapshot.particles.typeid[n]         = state.bbTypes[n]
                snapshot.particles.mass[n]           = BBlockTypeList[bbN].mass
                snapshot.particles.moment_inertia[n] = BBlockTypeList[bbN].moment_inertia
                all_names.append(SystDef.GetParticleTypeStrings(bbN))

        system = hoomd.init.read_snapshot(snapshot)

        # Only unique types are listed
        unique_types = np.unique(all_names)

        for ut in unique_types:
                system.particles.types.add(ut)

        # Creates a list of objects rigid body
        rigid = hoomd.md.constrain.rigid()

        for bbN in state.bbTypes:

                # Populate the rigid body: to the dumb particle attach all the others of types and positions
                rigid.set_param(COM_names[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

        # Creates the rigid body object
        rigid.create_bodies()
        rigid.validate_bodies()

        # Creates the list to compute interactions
        nl = hoomd.md.nlist.cell()

        # Add the dumb center of mass particle to the particles list
        alltypes = COM_names+unique_types.tolist()

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


        # Groups particles
        rigid = hoomd.group.rigid_center();

        # Number of rigid bodies
        N = len(rigid)
        #assert N==params.N

        # Applies integration scheme to the group
        fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=1, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-5, wtol=1e-5, Etol=1e-11, min_steps=500, group=rigid, aniso=True)

        #fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-7, wtol=1e-7, Etol=1e-07, min_steps=10, group=rigid, aniso=True)

        #hoomd.md.integrate.mode_standard(dt=params.dT);

        #nve = hoomd.md.integrate.nve(group=rigid)
        #hoomd.md.integrate.brownian(group=rigid, kT=0.0, seed=params.seed_brown)

        # Save total potential energy
        #hoomd.analyze.log(filename='%s_Data%d' % (params.fnbase,str_j), quantities=['time','potential_energy'], period=params.Trec_data, header_prefix='#', overwrite=True, phase=0)

        # Saves trajectories in a gsd file
        # hoomd.dump.gsd("%s_EMin.gsd" % params.fnbase,
        #                period=params.Trec_traj,
        #                group=hoomd.group.all(),
        #                #group=rigid,
        #                overwrite=True,
        #                dynamic =['attribute']
        # );


        # callback function to save the total net force of the system on a file, period equal to the saving period of the energy
        #ForceFile = open('%sForce%d' % (params.fnbase,str_j),'w')

        # def force(time_step):
        #         f = 0
        #         for p in system.particles:
        #                 f += np.linalg.norm(p.net_force)
        #         #TF = fire.has_converged()
        #         #ForceFile.write("{} {} {} {} {}\n".format(time_step,f[0],f[1],f[2],TF))
        #         ForceFile.write("{} {}\n".format(time_step,f))
        #         ForceFile.flush()
        #         #print(fire.has_converged())

        # Run like hell

        counter = 0

        while not(fire.has_converged()):
                if counter>1000:
                        break
                #hoomd.run(params.Ttot, callback_period=params.Trec_data, callback=force)
                hoomd.run(params.Ttot)
                counter += 1

        #hoomd.run(1000, callback_period=params.Trec_data, callback=force)

        if not fire.has_converged():
                print("\n")
                print("WARNING: convergence not reached!")
                print("\n")
        else:
                # I am happy
                print("\n")
                print("so far so good")
                print("\n")

        snap_curr = system.take_snapshot()

        for i in range(len(state.bbTypes)):
                state.positions[i]=snap_curr.particles.position[i]
                state.orientations[i]=snap_curr.particles.orientation[i]

        # ForceFile.close()




def FireSingleEnMin_new(params,SystDef,state,str_j):

        PartTypes  = SystDef.particleTypes
        BBlockTypeList = SystDef.buildingBlockTypeList

        N_state = len(state.bbTypes)

        COM_names = []

        for tn in state.bbTypes:
                COM_names.append("C%s" % tn)

        hoomd.context.initialize("");

        snapshot = hoomd.data.make_snapshot(N=N_state,
                                            box=hoomd.data.boxdim(L=10),
                                            particle_types=COM_names,
                                    )

        all_names = []

        for n,bbN in enumerate(state.bbTypes):
                snapshot.particles.position[n]       = state.positions[n]
                snapshot.particles.orientation[n]    = state.orientations[n]
                snapshot.particles.typeid[n]         = state.bbTypes[n]
                snapshot.particles.mass[n]           = BBlockTypeList[bbN].mass
                snapshot.particles.moment_inertia[n] = BBlockTypeList[bbN].moment_inertia
                all_names.append(SystDef.GetParticleTypeStrings(bbN))

        system = hoomd.init.read_snapshot(snapshot)

        # Only unique types are listed
        unique_types = np.unique(all_names)

        for ut in unique_types:
                system.particles.types.add(ut)

        # Creates a list of objects rigid body
        rigid = hoomd.md.constrain.rigid()

        for bbN in state.bbTypes:

                # Populate the rigid body: to the dumb particle attach all the others of types and positions
                rigid.set_param(COM_names[bbN], types = SystDef.GetParticleTypeStrings(bbN).tolist(), positions = BBlockTypeList[bbN].positions)

        # Creates the rigid body object
        rigid.create_bodies()
        rigid.validate_bodies()

        # Creates the list to compute interactions
        nl = hoomd.md.nlist.cell()

        # Add the dumb center of mass particle to the particles list
        alltypes = COM_names+unique_types.tolist()

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


        # Groups particles
        rigid_gr = hoomd.group.rigid_center()

        # Number of rigid bodies
        N = len(rigid_gr)
        #assert N==params.N

        # Applies integration scheme to the group
        fire = hoomd.md.integrate.mode_minimize_fire(dt=params.dT, Nmin=1, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol=1e-5, wtol=1e-5, Etol=1e-11, min_steps=500, group=rigid_gr, aniso=True)

        # Run like hell

        counter = 0

        while not(fire.has_converged()):
                if counter>1000:
                        break
                #hoomd.run(params.Ttot, callback_period=params.Trec_data, callback=force)
                hoomd.run(params.Ttot)
                counter += 1

        #hoomd.run(1000, callback_period=params.Trec_data, callback=force)

        if not fire.has_converged():
                print("\n")
                print("WARNING: convergence not reached!")
                print("\n")
        else:
                # I am happy
                print("\n")
                print("so far so good")
                print("\n")

        snap_curr = system.take_snapshot()

        for i in range(len(state.bbTypes)):
                state.positions[i]=snap_curr.particles.position[i]
                state.orientations[i]=snap_curr.particles.orientation[i]

        # ForceFile.close()






def parserFunc():
        parser = argparse.ArgumentParser(description='Run a simulation.')
        parser.add_argument('-i', '--input_name', type=str,            default='test', help='input file')
        parser.add_argument('-D', '--morse_D0',   type=float,          default=3.,     help='morse min')
        parser.add_argument('-a', '--morse_a',    type=float,          default=5.,     help='morse strength')
        parser.add_argument('-c', '--conc',       type=float,          default=0.001,  help='concentration')
        parser.add_argument('-s', '--string',     type=str,            default='test', help='output file name')
        parser.add_argument('-t', '--testing',    action="store_true", default=False,  help='put the simulation in testing mode')

        return parser

#parserFunc()
#RunEnergyMinimization(params)
