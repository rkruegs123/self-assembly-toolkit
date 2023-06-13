#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:15:48 2019
"""

import numpy as np
from potentials import *
from classDefinitions import *
import transformations
#import rowan

def gradientForParticlePair(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos, dE):
    #particle1RelativePos = relative position of particle 1 compared to building block center of mass
    #bb1Pos = position and orientation of building block 1
    #returns a length-12 vector: how energy changes as fxn of positions & orientations of the 2 building blocks.
    #dE is dE/dr

    particle1Pos = getPosFromRelativePos(particle1RelativePos, bb1Pos)
    particle2Pos = getPosFromRelativePos(particle2RelativePos, bb2Pos)

    dr = dr_dalpha(particle1Pos, particle2Pos) # How separation (r) changes as a function of positions of (interacting) particles.
    dalpha = dalpha_dmu(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos)
    # How positions of particles change as a function of positions & orientations of building blocks

    drdmu = np.dot(dr, dalpha)

    return(dE * drdmu)


def dynamicalMatrixForParticlePair(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos, dE, d2E):
    #particle1RelativePos = relative position of particle 1 compared to building block center of mass
    #bb1Pos = position and orientation of building block 1
    #returns a 12x12 array

    particle1Pos = getPosFromRelativePos(particle1RelativePos, bb1Pos)
    particle2Pos = getPosFromRelativePos(particle2RelativePos, bb2Pos)

    dr = dr_dalpha(particle1Pos, particle2Pos)
    d2r = d2r_dalpha_dbeta(particle1Pos, particle2Pos)
    dalpha = dalpha_dmu(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos)
    d2alpha = d2alpha_dmu_dnu(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos)

    drdmu = np.dot(dr, dalpha)

    return(d2E * np.outer(drdmu, drdmu) +
           dE * np.matmul(np.matmul(np.transpose(dalpha), d2r), dalpha) + #np.matmul(a,b) here is identical to np.tensordot(a,b,axes=1)
           dE * np.tensordot(dr, d2alpha, axes = 1)) #here it's important to use tensordot for the shape to work out


def getPosFromRelativePos(particleRelativePos, bbPos):
    #Gets the position in 3D space of a particle given the building block
    #center of mass position and orientation and the particle's relative position
    #Using Mathematica's default definition of EulerMatrix
    x0, y0, z0 = particleRelativePos
    xbar, ybar, zbar, abar, bbar, gbar = bbPos

    sa = np.sin(abar); ca = np.cos(abar)
    sb = np.sin(bbar); cb = np.cos(bbar)
    sg = np.sin(gbar); cg = np.cos(gbar)

    x = xbar + z0*ca*sb + y0*(-cg*sa - ca*cb*sg) + x0*(ca*cb*cg - sa*sg)
    y = ybar + z0*sa*sb + x0*(cb*cg*sa + ca*sg) + y0*(ca*cg - cb*sa*sg)
    z = zbar + z0*cb - x0*cg*sb + y0*sb*sg

    return(np.array([x,y,z]))


def dr_dalpha(particle1Pos, particle2Pos):
    #returns dr/dalpha where alpha is x1, y1, z1, x2, y2, z2
    #returns length 6 vector
    x1,y1,z1 = particle1Pos
    x2,y2,z2 = particle2Pos
    r = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return(np.array([(x1-x2)/r, (y1-y2)/r, (z1-z2)/r, (x2-x1)/r, (y2-y1)/r, (z2-z1)/r]))

def d2r_dalpha_dbeta(particle1Pos, particle2Pos):
    #returns d^2r/dalpha dbeta where alpha is x1, y1, z1, x2, y2, z2 and so is beta
    #returns 6x6 array
    x1,y1,z1 = particle1Pos
    x2,y2,z2 = particle2Pos
    r = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    return(np.array([[(y1 - y2)**2 + (z1 - z2)**2, -(x1 - x2)*(y1 - y2), -(x1 - x2)*(z1 - z2), -(y1 - y2)**2 - (z1 - z2)**2, (x1 - x2)*(y1 - y2), (x1 - x2)*(z1 - z2)],
                   [-(x1 - x2)*(y1 - y2), (x1 - x2)**2 + (z1 - z2)**2, -(y1 - y2)*(z1 - z2), (x1 - x2)*(y1 - y2), -(x1 - x2)**2 - (z1 - z2)**2, (y1 - y2)*(z1 - z2)],
                   [-(x1 - x2)*(z1 - z2), -(y1 - y2)*(z1 - z2), (x1 - x2)**2 + (y1 - y2)**2, (x1 - x2)*(z1 - z2), (y1 - y2)*(z1 - z2), -(x1 - x2)**2 - (y1 - y2)**2],
                   [-(y1 - y2)**2 - (z1 - z2)**2, (x1 - x2)*(y1 - y2), (x1 - x2)*(z1 - z2), (y1 - y2)**2 + (z1 - z2)**2, -(x1 - x2)*(y1 - y2), -(x1 - x2)*(z1 - z2)],
                   [(x1 - x2)*(y1 - y2), -(x1 - x2)**2 - (z1 - z2)**2, (y1 - y2)*(z1 - z2), -(x1 - x2)*(y1 - y2), (x1 - x2)**2 + (z1 - z2)**2, -(y1 - y2)*(z1 - z2)],
                   [(x1 - x2)*(z1 - z2), (y1 - y2)*(z1 - z2), -(x1 - x2)**2 - (y1 - y2)**2, -(x1 - x2)*(z1 - z2), -(y1 - y2)*(z1 - z2), (x1 - x2)**2 + (y1 - y2)**2]])
                    /r**3)


def dalpha_dmu(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos):
    #returns 6x12 array
    #This has a simple block form because x1 = x1Bar + f(abar1, bbar1,gbar1), etc.

    dxdEulerAngles1 = dx_dEulerAngles(particle1RelativePos, bb1Pos)
    dxdEulerAngles2 = dx_dEulerAngles(particle2RelativePos, bb2Pos)

    return(np.block([[np.identity(3), dxdEulerAngles1, np.zeros((3,6))],[np.zeros((3,6)),np.identity(3), dxdEulerAngles2]]))

def dx_dEulerAngles(particleRelativePos, bbPos):
#    dxdEulerAngles is a 3x3 matrix of dx/dalphaBar, dx/dbetaBar, dx/dgammaBar
#                                      dy/dalphaBar, dy/dbetaBar, dy/dgammaBar
#                                      dz/dalphaBar, dz/dbetaBar, dz/dgammaBar

    x0, y0, z0 = particleRelativePos
    xbar, ybar, zbar, abar, bbar, gbar = bbPos

    sa = np.sin(abar); ca = np.cos(abar)
    sb = np.sin(bbar); cb = np.cos(bbar)
    sg = np.sin(gbar); cg = np.cos(gbar)

    #Using Mathematica's default definition of EulerMatrix

    return(np.array([[-ca*(y0*cg + x0*sg) - sa*(z0*sb + cb*(x0*cg - y0*sg)),
                      ca*(z0*cb + sb*(-x0*cg + y0*sg)),
                      -ca*cb*(y0*cg + x0*sg) + sa*(-x0*cg + y0*sg)],
                     [-sa*(y0*cg + x0*sg) + ca*(z0*sb + cb*(x0*cg - y0*sg)),
                      sa*(z0*cb + sb*(-x0*cg + y0*sg)),
                      -cb*sa*(y0*cg + x0*sg) + ca*(x0*cg - y0*sg)],
                     [0, -z0*sb + cb*(-x0*cg + y0*sg), sb*(y0*cg + x0*sg)]]))

def d2alpha_dmu_dnu(particle1RelativePos, particle2RelativePos, bb1Pos, bb2Pos):
    #returns 6x12x12 array. All zeros except for two 3x3x3 blocks of d^2x1/dalphaBar1dalphaBar1, ... d^2z1/dgammaBar1dgammaBar1.


    d2xdEulerAngles1 = d2x_dEulerAngles2(particle1RelativePos, bb1Pos)
    d2xdEulerAngles2 = d2x_dEulerAngles2(particle2RelativePos, bb2Pos)

    return(np.block([[[np.zeros((3,3,12))],
                      [np.zeros((3,3,3)),d2xdEulerAngles1, np.zeros((3,3,6))],
                      [np.zeros((3,6,12))]],
                      [[np.zeros((3,6,12))],
                      [np.zeros((3,3,12))],
                      [np.zeros((3,3,3)), np.zeros((3,3,6)),d2xdEulerAngles2]]]))

def d2x_dEulerAngles2(particleRelativePos, bbPos):
    #a 3x3x3 block of d^2x/dalphaBar_dalphaBar, all the way to d^2x/dgammaBar_dgammaBar
    #The first index is x,y,z; the second and third are alpha, beta, gamma

    #Using Mathematica's default definition of EulerMatrix

    x0, y0, z0 = particleRelativePos
    xbar, ybar, zbar, abar, bbar, gbar = bbPos

    sa = np.sin(abar); ca = np.cos(abar)
    sb = np.sin(bbar); cb = np.cos(bbar)
    sg = np.sin(gbar); cg = np.cos(gbar)

    return(np.array([[[sa*(y0*cg + x0*sg) - ca*(z0*sb + cb*(x0*cg - y0*sg)), -sa*(z0*cb + sb*(-x0*cg + y0*sg)), cb*sa*(y0*cg + x0*sg) + ca*(-x0*cg + y0*sg)],
                      [-sa*(z0*cb + sb*(-x0*cg + y0*sg)), -ca*(z0*sb + cb*(x0*cg - y0*sg)), ca*sb*(y0*cg + x0*sg)],
                      [cb*sa*(y0*cg + x0*sg) + ca*(-x0*cg + y0*sg), ca*sb*(y0*cg + x0*sg), sa*(y0*cg + x0*sg) + ca*cb*(-x0*cg + y0*sg)]],
                     [[-ca*(y0*cg + x0*sg) - sa*(z0*sb + cb*(x0*cg - y0*sg)), ca*(z0*cb + sb*(-x0*cg + y0*sg)), -ca*cb*(y0*cg + x0*sg) + sa*(-x0*cg + y0*sg)],
                      [ca*(z0*cb + sb*(-x0*cg + y0*sg)), -sa*(z0*sb + cb*(x0*cg - y0*sg)), sa*sb*(y0*cg + x0*sg)],
                      [-ca*cb*(y0*cg + x0*sg) + sa*(-x0*cg + y0*sg), sa*sb*(y0*cg + x0*sg), -ca*(y0*cg + x0*sg) + cb*sa*(-x0*cg + y0*sg)]],
                     [[0,0,0], [0,-z0*cb + sb*(x0*cg - y0*sg), cb*(y0*cg + x0*sg)],
                      [0,cb*(y0*cg + x0*sg),sb*(x0*cg - y0*sg)]]]))




def mainGradientCalc(sysDef, state):
# =============================================================================
# =============================================================================

    buildingBlockPositions = state.positions
    #buildingBlockOrientations = euler_angles
    buildingBlockOrientations = np.array([ transformations.euler_from_quaternion(q, axes='rzyz') for q in state.orientations ])
    #buildingBlockOrientations = np.array([ rowan.to_euler(q, convention='zyz') for q in np.array(rowan.normalize(state.orientations)) ])
    buildingBlockTypes = state.bbTypes

    M = len(buildingBlockTypes) #number of building blocks in structure
    G = np.zeros(6*M) #The gradient vector

    totalE = 0

    for bbIndex1 in range(M):
        bb1Type = buildingBlockTypes[bbIndex1]
        bb1Pos = np.append(buildingBlockPositions[bbIndex1][:], buildingBlockOrientations[bbIndex1][:]) #bb1Pos is both position and orientation

        particlePositions1 = sysDef.buildingBlockTypeList[bb1Type].positions #an N1x3 array, where N1 = # of particles in bb1
        #particlePositions is just a list of [x,y,z] coordinates relative to the center of mass of the building block of type bb1Type
        particleTypes1 = sysDef.buildingBlockTypeList[bb1Type].typeids #an N1-length np array (vector)

        numParticles1 = len(particleTypes1)

        for bbIndex2 in range(bbIndex1+1,M): #can't interact within a building block
            bb2Type = buildingBlockTypes[bbIndex2]
            bb2Pos = np.append(buildingBlockPositions[bbIndex2][:], buildingBlockOrientations[bbIndex2][:]) #bb2Pos is both position and orientation

            particlePositions2 = sysDef.buildingBlockTypeList[bb2Type].positions #an N2x3 array, where N2 = # of particles in bb2
            #particlePositions is just a list of [x,y,z] coordinates relative to the center of mass of the building block of type bb1Type
            particleTypes2 = sysDef.buildingBlockTypeList[bb2Type].typeids #an N1-length np array (vector)

            numParticles2 = len(particleTypes2)

            for particle1Index in range(numParticles1):
                for particle2Index in range(numParticles2):

                    particle1Type = particleTypes1[particle1Index]
                    particle2Type = particleTypes2[particle2Index]

                    interactionParams = sysDef.interactions.matrix[particle1Type][particle2Type] #a list of parameters

                    # interactionPotentials is a list that could be empty, or could have one or multiple potentials these
                    # particles interact between

                    if any(interactionParams): #if there is some sort of interaction; otherwise, don't bother computing separation
                        particle1RelativePos = particlePositions1[particle1Index][:]
                        particle2RelativePos = particlePositions2[particle2Index][:]

                        particle1Pos = getPosFromRelativePos(particle1RelativePos, bb1Pos)
                        particle2Pos = getPosFromRelativePos(particle2RelativePos, bb2Pos)

                        separation = np.linalg.norm(particle1Pos - particle2Pos)

                        dE = 0
                        for potCounter in range(len(interactionParams)):
                            if interactionParams[potCounter] is not None:
                                pot = sysDef.interactions.potentials[potCounter]
                                params = interactionParams[potCounter]

                                totalE += pot.E(separation, rmin = params.rmin, rmax = params.rmax, **params.coeff)
                                dE += pot.dE(separation, rmin = params.rmin, rmax = params.rmax, **params.coeff)

                        gradient12 = gradientForParticlePair(particle1RelativePos, particle2RelativePos,
                                                                             bb1Pos, bb2Pos, dE)

                        #add this to H in the right spots
                        G[6*bbIndex1:6*(bbIndex1+1)] += gradient12[:6]
                        G[6*bbIndex2:6*(bbIndex2+1)] += gradient12[6:]

    return(G, totalE)




def mainDynamicalMatrixCalc(sysDef, state):
# =============================================================================
#    XXX MAKE SURE WHEN YOU MINIMIZE THE ENERGIES TO GET THE BUILDING BLOCK POSITIONS, YOU CENTER THEM
#    TO AVOID ISSUES WITH PERIODIC B.C.s

#    Dynamical matrix should have (for a dimer) 6 zero modes, none of the evals should be negative,
#    blocks that correspond to x,y,zs should be symmetric; diagonal part of positional matrix should be
#    negative the sum of the off diagonals.
#    non-zero modes should correspond to expected motions of the building blocks
#    and finally, check with Mathematica

#    If you pick a random orientation in phase space and move the bbs a bit in that direction (U)
#    it should agree with 1/2 U^T H U
# =============================================================================


    buildingBlockPositions = state.positions
    #buildingBlockOrientations = euler_angles
    buildingBlockOrientations = np.array([ transformations.euler_from_quaternion(q, axes='rzyz') for q in state.orientations ])
    #buildingBlockOrientations = np.array([ rowan.to_euler(q, convention='zyz') for q in rowan.normalize(state.orientations) ])
    buildingBlockTypes = state.bbTypes


    M = len(buildingBlockTypes) #number of building blocks in structure
    H = np.zeros((6*M,6*M)) #The Hessian of the dynamical matrix
    dynamicalMatrix = np.zeros((6*M,6*M))

    totalE = 0

    for bbIndex1 in range(M):
        # type of building block, for now it is always 0, buildingBlockTypes is a list of [0,0]
        bb1Type = buildingBlockTypes[bbIndex1]
        bb1Pos = np.append(buildingBlockPositions[bbIndex1][:], buildingBlockOrientations[bbIndex1][:]) #bb1Pos is both position and orientation

        particlePositions1 = sysDef.buildingBlockTypeList[bb1Type].positions #an N1x3 array, where N1 = # of particles in bb1
        #particlePositions is just a list of [x,y,z] coordinates relative to the center of mass of the building block of type bb1Type

        particleTypes1 = sysDef.buildingBlockTypeList[bb1Type].typeids #an N1-length np array (vector)
        bbMass1 = sysDef.buildingBlockTypeList[bb1Type].mass #a scalar
        bbMomentOfInertia1 = sysDef.buildingBlockTypeList[bb1Type].moment_inertia #a length-3 vector

        numParticles1 = len(particleTypes1)

        for bbIndex2 in range(bbIndex1+1,M): #can't interact within a building block
            bb2Type = buildingBlockTypes[bbIndex2]
            bb2Pos = np.append(buildingBlockPositions[bbIndex2][:], buildingBlockOrientations[bbIndex2][:]) #bb2Pos is both position and orientation

            particlePositions2 = sysDef.buildingBlockTypeList[bb2Type].positions #an N2x3 array, where N2 = # of particles in bb2
            #particlePositions is just a list of [x,y,z] coordinates relative to the center of mass of the building block of type bb1Type

            particleTypes2 = sysDef.buildingBlockTypeList[bb2Type].typeids #an N1-length np array (vector)
            bbMass2 = sysDef.buildingBlockTypeList[bb2Type].mass #a scalar
            bbMomentOfInertia2 = sysDef.buildingBlockTypeList[bb2Type].moment_inertia #a length-3 vector

            numParticles2 = len(particleTypes2)

            sqrtMassesAndIs = np.sqrt([bbMass1]*3 + list(bbMomentOfInertia1) + [bbMass2]*3 + list(bbMomentOfInertia2))
            # Define matrix that converts from the 12x12 Hessian to the dynamical matrix when you divide by it
            hessianToDynamicalMatrix = np.outer(sqrtMassesAndIs, sqrtMassesAndIs)


            for particle1Index in range(numParticles1):
                for particle2Index in range(numParticles2):

                    particle1Type = particleTypes1[particle1Index]
                    particle2Type = particleTypes2[particle2Index]

                    interactionParams = sysDef.interactions.matrix[particle1Type][particle2Type] #a list of parameters

                    #if sysDef.particleTypes[particle1Type].typeString[1:] == sysDef.particleTypes[particle2Type].typeString[1:]:
                    #    print(sysDef.particleTypes[particle1Type].typeString,sysDef.particleTypes[particle2Type].typeString,interactionParams)

                    # interactionPotentials is a list that could be empty, or could have one or multiple potentials these
                    # particles interact between

                    if any(interactionParams): #if there is some sort of interaction; otherwise, don't bother computing separation
                        particle1RelativePos = particlePositions1[particle1Index][:]
                        particle2RelativePos = particlePositions2[particle2Index][:]

                        particle1Pos = getPosFromRelativePos(particle1RelativePos, bb1Pos)
                        particle2Pos = getPosFromRelativePos(particle2RelativePos, bb2Pos)

                        separation = np.linalg.norm(particle1Pos - particle2Pos)

                        dE = 0; d2E = 0
                        for potCounter in range(len(interactionParams)):
                            if interactionParams[potCounter] is not None:
                                pot = sysDef.interactions.potentials[potCounter]
                                params = interactionParams[potCounter]

                                totalE += pot.E(separation, rmin = params.rmin, rmax = params.rmax, **params.coeff)
                                dE += pot.dE(separation, rmin = params.rmin, rmax = params.rmax, **params.coeff)
                                d2E += pot.d2E(separation, rmin = params.rmin, rmax = params.rmax, **params.coeff)

                        hessian1212 = dynamicalMatrixForParticlePair(particle1RelativePos, particle2RelativePos,
                                                                             bb1Pos, bb2Pos, dE, d2E)


                        #add this to H in the right spots
                        H[6*bbIndex1:6*(bbIndex1+1),6*bbIndex1:6*(bbIndex1+1)] += hessian1212[:6,:6]
                        H[6*bbIndex1:6*(bbIndex1+1),6*bbIndex2:6*(bbIndex2+1)] += hessian1212[:6,6:]
                        H[6*bbIndex2:6*(bbIndex2+1),6*bbIndex1:6*(bbIndex1+1)] += hessian1212[6:,:6]
                        H[6*bbIndex2:6*(bbIndex2+1),6*bbIndex2:6*(bbIndex2+1)] += hessian1212[6:,6:]


                        dynamicalMatrix1212 = hessian1212 / hessianToDynamicalMatrix

                        dynamicalMatrix[6*bbIndex1:6*(bbIndex1+1),6*bbIndex1:6*(bbIndex1+1)] += dynamicalMatrix1212[:6,:6]
                        dynamicalMatrix[6*bbIndex1:6*(bbIndex1+1),6*bbIndex2:6*(bbIndex2+1)] += dynamicalMatrix1212[:6,6:]
                        dynamicalMatrix[6*bbIndex2:6*(bbIndex2+1),6*bbIndex1:6*(bbIndex1+1)] += dynamicalMatrix1212[6:,:6]
                        dynamicalMatrix[6*bbIndex2:6*(bbIndex2+1),6*bbIndex2:6*(bbIndex2+1)] += dynamicalMatrix1212[6:,6:]

    #print("tot E",totalE)
    return(H, dynamicalMatrix, totalE)

# =============================================================================
#  Checks
# =============================================================================

def check_d2rMatrix(): #check for arbitrarily chosen positions
    particle1Pos = [1,2,3]
    particle2Pos = [5,7,11]
    eps = 1e-8 #to define rounding errors
    r = np.sqrt(np.dot(np.array(particle1Pos)-np.array(particle2Pos),
                       np.array(particle1Pos)-np.array(particle2Pos)))
    realAns = np.array([[89, -20, -32, -89, 20, 32], [-20, 80, -40, 20, -80, 40], [-32, -40, 41, 32, 40, -41],
                        [-89, 20, 32, 89, -20, -32], [20, -80, 40, -20, 80, -40], [32, 40, -41, -32, -40, 41]])/r**3
    #THIS IS IN THE ORDER x1, y1, z1, x2, y2, z2
    if (abs(d2r_dalpha_dbeta(particle1Pos, particle2Pos) - realAns) > eps).any():
        return(False)
    else:
        return(True)
