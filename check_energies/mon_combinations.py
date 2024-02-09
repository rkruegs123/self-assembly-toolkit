import matplotlib.pyplot as plt
import pdb
import numpy as onp
import jax.numpy as jnp
from jax import  vmap
import itertools

from jax.config import config
config.update("jax_enable_x64", True)



monomers = {
    'A': [1, 0, 2],
    'B': [3, 0, 4],
    'C': [5, 0, 6],
}


monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}

all_monomers = {**monomers, **monomers_prime}

def flatten_and_compare(struct1, struct2):
    struct1_nums = sum([all_monomers[mon] for mon in struct1], [])
    struct2_nums = sum([all_monomers[mon] for mon in struct2], [])
    return struct1_nums == struct2_nums[::-1]

all_combinations = []
for r in range(1, 4):  
    all_combinations.extend(itertools.combinations_with_replacement(all_monomers.keys(), r))


filtered_combinations = []
for comb in all_combinations:
    if all(not flatten_and_compare(comb, existing_comb) for existing_comb in filtered_combinations):
        filtered_combinations.append(comb)
        
mon_list = []
dimer_list = []
trimer_list = []

for comb in filtered_combinations:
    if len(comb) == 1:  # Monomer
        mon_list.append(comb)
    elif len(comb) == 2:  # Dimer
        dimer_list.append(comb)
    elif len(comb) == 3:  # Trimer
        trimer_list.append(comb)
        
def count_monomers(combinations, monomer_name):
    """
    Count how many times a specific monomer and its mirrored version appear in each combination.
    """
    monomer_counts = []
    for comb in combinations:
        count = 0        
        for mon in comb:
            if mon == monomer_name or mon == f"{monomer_name}'":  
                count += 1
        
        monomer_counts.append(count)
    
    return monomer_counts

A_dimer_counts = count_monomers(dimer_list, 'A')
A_trimer_counts = count_monomers(trimer_list, 'A')
B_dimer_counts = count_monomers(dimer_list, 'B')
B_trimer_counts = count_monomers(trimer_list, 'B')
C_dimer_counts = count_monomers(dimer_list, 'C')
C_trimer_counts = count_monomers(trimer_list, 'C')
        
       
def combination_to_string(comb):
    return ' '.join(comb)

mon_list = [combination_to_string(comb) for comb in mon_list]
dimer_list = [combination_to_string(comb) for comb in dimer_list]
trimer_list = [combination_to_string(comb) for comb in trimer_list]

def get_numeric_combination(comb_str):

    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])    
    return numeric_combination


mon_pc_species = [get_numeric_combination(mon) for mon in mon_list]
dimer_pc_specie = [get_numeric_combination(dimer) for dimer in dimer_list]
trimer_pc_species = [get_numeric_combination(trimer) for trimer in trimer_list]
