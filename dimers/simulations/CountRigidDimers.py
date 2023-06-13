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

The list is then used to count the repeats of each structure.

"""

from __future__ import division
import sys
import gsd.fl
import gsd.hoomd
import hoomd
import hoomd.md
import math
import numpy as np
import random
from AnalysisFunctions import *





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



# always read those parameters from the command line arguments
# morse_a is usually default at 5
params.morse_D0 = args.morse_D0
params.concentration = args.conc
#params.morse_a = args.morse_a

# input name
basename  = 'temp_results/%s_Traj' % args.input_name
inputname = basename+'.gsd'



# file where the results are going to be stored
# it is organized in the following way:

# t=current_timestep
# counts | size of adj matrix | adj matrix | positions | quaternions *of structure 1*
# counts | size of adj matrix | adj matrix | positions | quaternions *of structure 2*
# ...
listFilename = open('temp_results/%slist.txt' % args.input_name,'w')



# where everything happens:
# main loop over the snapshots
# # find the non-minimized structures
# # minimizes the energy for the non-monomers
# # recomputes the structures and counts them
CountClusters(inputname,params,listFilename)



# close the file
listFilename.close()
