import pickle
import jax.numpy as jnp
from tqdm import tqdm
import optax
from jax import jit, vmap, value_and_grad, lax, random


with open('log_z_all.pkl', 'rb') as f:
    log_z_all = pickle.load(f)

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_species_combinations('AB_species_test2.pkl')
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

def ofer(log_z_list, concs):
    log_z_mon = log_z_list[0:n]
    log_mon_conc = safe_log(concs[0:n])
    
    def loss_fn(log_concs_struc):
        def mon_loss_fn(mon_idx):
            mon_val = jnp.log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
            diff = mon_val - log_mon_conc[mon_idx]
            return jnp.abs(diff)

        def struc_loss_fn(struc_idx):
            log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

            def get_vcs_denom(mon_idx):
                n_sa = nper_structure[mon_idx][struc_idx]
                log_vca = jnp.log(V) + log_concs_struc[mon_idx]
                return n_sa * log_vca

            vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()

            log_zs = log_z_list[struc_idx]

            def get_z_denom(mon_idx):
                n_sa = nper_structure[mon_idx][struc_idx]
                log_zalpha = log_z_list[mon_idx]
                return n_sa * log_zalpha

            z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

            diff = log_vcs - vcs_denom - log_zs + z_denom
            return jnp.abs(diff)
        
        mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
        struc_loss = vmap(struc_loss_fn)(jnp.arange(n, tot_num_structures))
        tot_loss = struc_loss.sum() + mon_loss.sum()
        return tot_loss
    
    struc_concs_guess = jnp.full(tot_num_structures, safe_log(concs.sum() / tot_num_structures))
    
    optimizer = optax.adam(1e-2)
    params = struc_concs_guess
    opt_state = optimizer.init(params)
    grad_fn = jit(value_and_grad(loss_fn))

    n_iters = 5000

    @jit
    def scan_fn(opt_info, idx):
        struct_concs, opt_state = opt_info
        loss, grads = grad_fn(struct_concs)
        updates, opt_state = optimizer.update(grads, opt_state)
        struct_concs = optax.apply_updates(struct_concs, updates)
        return (struct_concs, opt_state), (loss, struct_concs)

    fin_opt_info, (losses, iter_concs) = lax.scan(scan_fn, (params, opt_state), jnp.arange(n_iters))
    fin_log_concs, fin_opt_state = fin_opt_info
    
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / fin_concs.sum()
    log_yields = jnp.log(yields)

    return -log_yields[-1], (losses, iter_concs)

conc_A = 0.01
conc_B = 0.02
m_conc = jnp.array([conc_A, conc_B])

init_log_mon_concs = safe_log(jnp.array([conc_A, conc_B]))
init_struc_concs = jnp.full(tot_num_structures, safe_log(m_conc.sum() / tot_num_structures))

my_fn = lambda some_log_concs: ofer(log_z_all, some_log_concs)
our_grad_fn = value_and_grad(my_fn, has_aux=True)
our_grad_fn = jit(our_grad_fn)

lr = 1e-1
outer_optimizer = optax.adam(lr)
params = init_log_mon_concs
opt_state = outer_optimizer.init(params)

n_outer_iters = 10
for i in tqdm(range(n_outer_iters)):
    (val, (losses, iter_concs)), grads = our_grad_fn(params)
    print(f"Iteration {i+1}/{n_outer_iters}")
    print(f"Yield: {val}")
    print(f"Params: {params}")
    print(f"Converged to: {losses[-1]}")
    print(f"Gradients: {grads}")
    
    updates, opt_state = outer_optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)




    






