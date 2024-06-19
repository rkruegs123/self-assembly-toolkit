import pickle
import jax.numpy as jnp
from tqdm import tqdm
from jax import value_and_grad, grad
import optax

with open('log_z_all.pkl', 'rb') as f:
    log_z_list = pickle.load(f)

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('combinations_sigma.pkl')
tot_num_structures = data['mon_pc_species'].shape[0] + data['dimer_pc_species'].shape[0] + data['trimer_pc_species'].shape[0]
n = 2
V = 1.0

A_mon_counts = data['A_mon_counts']
A_dim_counts = data['A_dimer_counts']
A_trim_counts = data['A_trimer_counts']

B_mon_counts = data['B_mon_counts']
B_dim_counts = data['B_dimer_counts']
B_trim_counts = data['B_trimer_counts']

A_count = jnp.concatenate([A_mon_counts, A_trim_counts, A_dim_counts])
B_count = jnp.concatenate([B_mon_counts, B_trim_counts, B_dim_counts])
nper_structure = jnp.array([A_count, B_count])

conc_A = 0.07
conc_B = 0.07
init_m_conc = jnp.array([conc_A, conc_B])


def ofer(concs):

    m_conc = concs
    tot_conc = jnp.sum(m_conc)
    log_mon_conc = safe_log(m_conc)

    def mon_loss_fn(log_concs_struc, mon_idx):
        mon_val = jnp.log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
        return mon_val - log_mon_conc[mon_idx]

    def struc_loss_fn(log_concs_struc, struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        def get_vcs_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_vca = jnp.log(V) + log_concs_struc[mon_idx]
            return n_sa * log_vca

        vcs_denom = jnp.array([get_vcs_denom(mon_idx) for mon_idx in range(n)])
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = jnp.array([get_z_denom(mon_idx) for mon_idx in range(n)])

        return log_vcs - jnp.sum(vcs_denom) - log_zs + jnp.sum(z_denom)

    def combined_loss_fn(log_concs_struc):
        mon_losses = jnp.array([mon_loss_fn(log_concs_struc, mon_idx) for mon_idx in range(n)])
        struc_losses = jnp.array([struc_loss_fn(log_concs_struc, struc_idx) for struc_idx in range(n, tot_num_structures)])
        return jnp.concatenate((mon_losses, struc_losses))

    struc_concs_guess = jnp.full(tot_num_structures, safe_log(tot_conc / tot_num_structures))

    def loss_fn(log_concs_struc):
        return jnp.linalg.norm(combined_loss_fn(log_concs_struc))

    opt_init, opt_update = optax.adam(learning_rate=0.1)
    opt_state = opt_init(struc_concs_guess)

    gradient_norm_threshold = 0.5 # Define a threshold for convergence
    max_inner_iters = 100  # Maximum iterations per set
    max_sets = 10  # Maximum sets of iterations

    for _ in range(max_sets):
        for _ in range(max_inner_iters):
            loss_val, grads = value_and_grad(loss_fn)(struc_concs_guess)
            gradient_norm = jnp.linalg.norm(grads)
            if gradient_norm < gradient_norm_threshold:
                break
            updates, opt_state = opt_update(grads, opt_state)
            struc_concs_guess = optax.apply_updates(struc_concs_guess, updates)
        else:
            continue  # Only executed if the inner loop did NOT break
        break  # Only executed if the inner loop DID break

    fin_log_concs = struc_concs_guess
    fin_concs = jnp.exp(fin_log_concs)

    # Calculate yields
    yields = fin_concs / jnp.sum(fin_concs)
    log_yields = jnp.log(yields)
    target_yield = log_yields[-1]

    return target_yield


def ofer_grad_fn(concs):
    target_yield = ofer(concs)
    return - target_yield

def project(params):
    return jnp.clip(params, a_min=0.001)

our_grad_fn = grad(ofer_grad_fn)


lr = 1e-3
outer_optimizer = optax.adam(lr)
params = init_m_conc
opt_state = outer_optimizer.init(params)

n_outer_iters = 10 # Increase number of iterations for optimization
for i in tqdm(range(n_outer_iters)):
    grads = our_grad_fn(params)


    updates, opt_state = outer_optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    params = project(params)

    print(f"Iteration {i+1}/{n_outer_iters}")
    print(f"Params: {params}")
    print(f"Gradients: {grads}")


final_params = params
final_target_yield = ofer(final_params)
print(f"Final Optimized Parameters (monomer concentrations): {final_params}")
print(f"Final Target Yield: {final_target_yield}")

"""


optimizer = optax.adam(1e-2)
params = struc_concs_guess
opt_state = optimizer.init(params)
grad_fn = jit(value_and_grad(loss_fn))

n_iters = 80000

losses = list()
all_grads = list()
for i in tqdm(range(n_iters)):
    loss, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    losses.append(loss)
    if i % 100 == 0:
        print(f"Iteration {i}: {loss}")
    all_grads.append(grads)

fin_log_concs = deepcopy(params)

fin_concs = jnp.exp(fin_log_concs)
yields = fin_concs / fin_concs.sum()
log_yields = jnp.log(yields)

target_yield = log_yields[-1]

pdb.set_trace()

print(target_yield)
"""