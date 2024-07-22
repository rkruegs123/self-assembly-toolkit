import itertools
import pickle
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor

# Define the types of monomers 
def create_monomers(num_monomers):
    monomers = {}
    for i in range(num_monomers):
        letter = chr(ord('A') + i)
        monomers[letter] = [2 * i + 1, 0, 2 * i + 2]
    return monomers

# Generate unique combinations of monomers of all sizes up to max_struc_size
def generate_combinations(monomers, max_struc_size):
    monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}
    all_monomers = {**monomers, **monomers_prime}
    
    all_combinations = []
    for r in range(1, max_struc_size + 1):
        all_combinations.extend(itertools.product(all_monomers.keys(), repeat=r))
    
    unique_combinations = []
    seen = set()
    for comb in all_combinations:
        mirrored_comb = tuple([k + "'" if k[-1] != "'" else k[:-1] for k in comb][::-1])
        if mirrored_comb not in seen:
            unique_combinations.append(comb)
            seen.add(comb)
    return unique_combinations, all_monomers

def combination_to_string(comb):
    return ' '.join(comb)

def get_numeric_combination(comb_str, all_monomers):
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination

# Count the occurrences of each monomer in the combinations
def count_monomers(combinations, monomer_name):
    monomer_counts = []
    for comb in combinations:
        count = sum(1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'")
        monomer_counts.append(count)
    return monomer_counts

def calculate_sigma(numeric_combination):
    length = len(numeric_combination)
    half_length = length // 2
    if length % 2 == 0 and numeric_combination[:half_length] == numeric_combination[half_length:][::-1]:
        return 6
    return 3

def process_combinations(unique_combinations, all_monomers, monomers, max_struc_size):
    species_dict = {}
    sigma_dict = {}
    counts = {}
    
    for size in range(1, max_struc_size + 1):
        structure_key = f'{size}_pc_species'
        sigma_key = f'{size}_sigma'
        current_combinations = [comb for comb in unique_combinations if len(comb) == size]
        species_dict[structure_key] = jnp.array([get_numeric_combination(combination_to_string(comb), all_monomers) for comb in current_combinations])
        
        if size % 2 == 0:
            sigma_dict[sigma_key] = jnp.array([calculate_sigma(get_numeric_combination(combination_to_string(comb), all_monomers)) for comb in current_combinations])
        else:
            sigma_dict[sigma_key] = jnp.array([3] * len(current_combinations))
        
        for letter in monomers.keys():
            count_key = f'{letter}_{size}_counts'
            counts[count_key] = jnp.array(count_monomers(current_combinations, letter))
    
    return {
        **species_dict,
        **sigma_dict,
        **counts
    }


def save_results(data):
    with open('species_combinations.pkl', 'wb') as f:
        pickle.dump(data, f)

def main(num_monomers, max_struc_size):
    monomers = create_monomers(num_monomers)
    unique_combinations, all_monomers = generate_combinations(monomers, max_struc_size)
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process_combinations, unique_combinations, all_monomers, monomers, max_struc_size)
        result = future.result()
    
    save_results(result)
    
    for size in range(1, max_struc_size + 1):
        sigma_key = f'{size}_sigma'
        if sigma_key in result:
            print(f"Sigma for size {size}: {result[sigma_key]}")
        else:
            print(f"No sigma values for size {size}")
    
    print("Species combinations and counts saved successfully.")
    print(sum(len(result[f'{i}_pc_species']) for i in range(1, max_struc_size + 1)))

if __name__ == "__main__":
    num_monomers = 3  # Fixme to the number of monomers
    max_struc_size = 3  # Fixme to the max size structure in the ensemble 
    main(num_monomers, max_struc_size)
