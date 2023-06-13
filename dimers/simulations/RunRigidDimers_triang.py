from __future__ import division
import random
import sys
import math
import numpy as np
from RunFunctions import *

import hoomd
import hoomd.md





# Initialize the simulation
hoomd.context.initialize("");





# Parameters
class CParams:
        # these are all class variables, they do not change in
        # different instances of the class CParams
        N = [9,9]
        concentration = 0.001
        ls = 10.

        Ttot = 1e9      # actually total number of steps. Ttot_real=Ttot
        Trec_data = 10000000
        Trec_traj = 10000000
        dT = 0.0001

        rep_A = 500.0
        rep_alpha = 2.5

        morse_D0 = 7.0
        morse_D0_r = 1.0 # 1.0
        morse_D0_g = 1.0 # 1.5
        morse_D0_b = 1.0 # 2.0

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

if(args.testing):
        params.N = [10,10]
        params.Ttot = 1e8
        params.Trec_data = 1000000
        params.Trec_traj = 1000000
        params.dT = 0.0001

params.fnbase = 'temp_results/%s_c%4.6f_D0%4.2f' % (args.string,args.conc,args.morse_D0)

params.concentration = args.conc
params.morse_D0 = args.morse_D0
#params.morse_a = args.morse_a

SystDef = InitializeSystem_dimers_tr(params)

RunSimulation(params,SystDef)
