import itertools
import pickle
import jax.numpy as jnp

# Define monomers and their mirrored versions
monomers = {
    'A': [1, 0, 2],
    'B': [3, 0, 4],
    'C': [5, 0, 6],
}

monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}
all_monomers = {**monomers, **monomers_prime}

# Function to compare structures
def flatten_and_compare(struct1, struct2):
    struct1_nums = sum([all_monomers[mon] for mon in struct1], [])
    struct2_nums = sum([all_monomers[mon] for mon in struct2], [])
    return struct1_nums == struct2_nums[::-1]

# Generate all combinations
all_combinations = []
for r in range(1, 4):
    all_combinations.extend(itertools.combinations_with_replacement(all_monomers.keys(), r))

# Filter combinations
filtered_combinations = []
for comb in all_combinations:
    if all(not flatten_and_compare(comb, existing_comb) for existing_comb in filtered_combinations):
        filtered_combinations.append(comb)

# Separate combinations by length
mon_list, dimer_list, trimer_list = [], [], []
for comb in filtered_combinations:
    if len(comb) == 1:
        mon_list.append(comb)
    elif len(comb) == 2:
        dimer_list.append(comb)
    elif len(comb) == 3:
        trimer_list.append(comb)

# Function to count monomers in combinations
def count_monomers(combinations, monomer_name):
    monomer_counts = []
    for comb in combinations:
        count = sum(1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'")
        monomer_counts.append(count)
    return monomer_counts

# Count monomers for each species type
A_mon_counts = jnp.array(count_monomers(mon_list, 'A'))
A_dimer_counts = jnp.array(count_monomers(dimer_list, 'A'))
A_trimer_counts = jnp.array(count_monomers(trimer_list, 'A'))
B_mon_counts = jnp.array(count_monomers(mon_list, 'B'))
B_dimer_counts = jnp.array(count_monomers(dimer_list, 'B'))
B_trimer_counts = jnp.array(count_monomers(trimer_list, 'B'))
C_mon_counts = jnp.array(count_monomers(mon_list, 'C'))
C_dimer_counts = jnp.array(count_monomers(dimer_list, 'C'))
C_trimer_counts = jnp.array(count_monomers(trimer_list, 'C'))

# Convert combinations to string and then to numeric combinations
def combination_to_string(comb):
    return ' '.join(comb)

def get_numeric_combination(comb_str):
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination

mon_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in mon_list])
dimer_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in dimer_list])
trimer_pc_species = jnp.array([get_numeric_combination(combination_to_string(comb)) for comb in trimer_list])

# Save all computed data in a single text file
with open('species_combination.txt', 'wb') as f:
    pickle.dump({
        'mon_pc_species': mon_pc_species,
        'dimer_pc_species': dimer_pc_species,
        'trimer_pc_species': trimer_pc_species,
        'A_mon_counts': A_mon_counts,
        'A_dimer_counts': A_dimer_counts,
        'A_trimer_counts': A_trimer_counts,
        'B_mon_counts': B_mon_counts,
        'B_dimer_counts': B_dimer_counts,
        'B_trimer_counts': B_trimer_counts,
        'C_mon_counts': C_mon_counts,
        'C_dimer_counts': C_dimer_counts,
        'C_trimer_counts': C_trimer_counts
    }, f)

print("Species combinations and counts saved successfully.")
