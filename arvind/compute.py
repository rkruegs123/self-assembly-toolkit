import pdb
import argparse
import numpy as onp
from jax import lax

from jax import random
from jax import jit, vmap
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)


TARGET = jnp.array([0, 1, 2])
DIMER_OFF_TARGETS = jnp.array([
    [0, 1],
    [0, 2],
    [1, 2]
])
DIMER_RB_SPECIES = jnp.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
N_SPECIES = len(set(onp.array(DIMER_RB_SPECIES.flatten())))


def distance(pos1, pos2):
    return jnp.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)


def morse(r, r0, eps, alpha):
    return eps * (1 - jnp.exp(-alpha * (r - r0)))**2
mapped_morse = vmap(morse, (0, None, 0, 0))

def target_energy_fn(R, args):
    distances = vmap(vmap(distance, (None, 0)), (0, None))(R, R)
    morse_vals = mapped_morse(distances, 0.0, args['morse_eps'], args['morse_alpha'])
    pdb.set_trace()

    # FIXME: mask along diagonal, sum, divide by 2
    pass

def dimer_off_target_energy_fn(args, dimer_off_target_idx):
    pass


def calc_target_z(args):

    dummy_R = jnp.array([
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],

    ], dtype=jnp.float64)


    beta = 1 / args['kt']
    energy = target_energy_fn(dummy_R, args)
    return jnp.exp(-beta * energy)


def calc_dimer_off_target_z(off_target, args):
    beta = 1 / args['kt']
    energy = dimer_off_target_energy_fn(off_target, args)
    return jnp.exp(-beta * energy)

def run(args):
    # morse_eps = args['morse_eps']
    # morse_alpha = args['morse_alpha']
    # concentrations = args['concentrations']


    target_z = calc_target_z(args)
    # dimer_ot_zs = vmap(calc_dimer_off_target_z, (0, None))(DIMER_OFF_TARGETS, args)

    # FIXME: compute yield via Zs and concentrations
    return



def get_argparse():
    parser = argparse.ArgumentParser(description="Compute the yield of an Arvind system")

    # System setup
    parser.add_argument("--default-conc", type=float,  default=0.001,
                        help="Default concentration of a given monomer")
    parser.add_argument("--kt", type=float,  default=1.0,
                        help="Temperature in kT")

    # Morse interaction
    parser.add_argument("--default-morse-eps", type=float,  default=10.0,
                        help="Default depth of Morse potential")
    parser.add_argument("--default-morse-alpha", type=float,  default=5.0,
                        help="Default width of Morse potential")

    return parser



if __name__ == "__main__":
    parser = get_argparse()
    default_args = vars(parser.parse_args())

    # Construct interaction matrices and conc. vector
    morse_eps = jnp.full((N_SPECIES, N_SPECIES), default_args['default_morse_eps'])
    morse_alpha = jnp.full((N_SPECIES, N_SPECIES), default_args['default_morse_alpha'])
    concentrations = jnp.full((N_SPECIES,), default_args['default_conc'])


    args = {
        "morse_eps": morse_eps,
        "morse_alpha": morse_alpha,
        "concentrations": concentrations,
        "kt": default_args['kt']
    }

    run(args)
