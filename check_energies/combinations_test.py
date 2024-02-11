import itertools
import pickle
import jax.numpy as jnp
import os


# Define monomers and their mirrored versions
monomers = {
    'A': [1, 0, 2]
    #'B': [3, 0, 4]
}

monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}
all_monomers = {**monomers, **monomers_prime}

# Generate all possible combinations with ordering for each length
all_combinations = []
for r in range(1, 4):
    all_combinations.extend(itertools.product(all_monomers.keys(), repeat=r))

# Function to determine if one combination is a mirrored version of another
def is_mirrored_duplicate(comb1, comb2):
    return comb1 == tuple([k + "'" if k[-1] != "'" else k[:-1] for k in comb2][::-1])

# Filter combinations to remove mirrored duplicates
unique_combinations = []
for comb in all_combinations:
    if not any(is_mirrored_duplicate(comb, existing_comb) for existing_comb in unique_combinations):
        unique_combinations.append(comb)

# Function to convert combinations to string and then to numeric combinations
def combination_to_string(comb):
    return ' '.join(comb)

def get_numeric_combination(comb_str):
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination

# Function to count monomers in combinations
def count_monomers(combinations, monomer_name):
    monomer_counts = []
    for comb in combinations:
        count = sum(1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'")
        monomer_counts.append(count)
    return monomer_counts

# Separate combinations by length and convert to numeric format
mon_list, dimer_list, trimer_list = [], [], []
for comb in unique_combinations:
    if len(comb) == 1:
        mon_list.append(comb)
    elif len(comb) == 2:
        dimer_list.append(comb)
    elif len(comb) == 3:
        trimer_list.append(comb)

mon_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in mon_list])
dimer_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in dimer_list])
trimer_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in trimer_list])

#  dynamically count monomers and save results
# Count monomers for each species type
A_mon_counts = jnp.array(count_monomers(mon_list, 'A'))
A_dimer_counts = jnp.array(count_monomers(dimer_list, 'A'))
A_trimer_counts = jnp.array(count_monomers(trimer_list, 'A'))



with open('species_test.txt', 'wb') as f:
    pickle.dump({
        'mon_pc_species': mon_pc_species,
        'dimer_pc_species': dimer_pc_species,
        'trimer_pc_species': trimer_pc_species,
        'A_mon_counts': A_mon_counts,
        'A_dimer_counts': A_dimer_counts,
        'A_trimer_counts': A_trimer_counts
    }, f)

print("Species combinations and counts saved successfully.")





