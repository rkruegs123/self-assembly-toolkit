
import pickle
from tqdm import tqdm
from copy import deepcopy
import jax.numpy as jnp
import optax
from jax import jit, vmap, value_and_grad

# Load data
with open('log_z_all_tri.pkl', 'rb') as f:
    log_z_list = pickle.load(f)

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

def load_species_combinations(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def finite_difference_grad(f, x, eps=1e-5):
    grad = jnp.zeros_like(x)
    for i in range(len(x)):
        x_eps = x.at[i].add(eps)
        f_eps = f(x_eps)
        grad = grad.at[i].set((f_eps - f(x)) / eps)
    return grad

data = load_species_combinations('ABC_species.pkl')
tot_num_structures = data['mon_pc_species'].shape[0] + data['dimer_pc_species'].shape[0] + data['trimer_pc_species'].shape[0]

n = 3
V = 100.0

A_mon_counts = data['A_mon_counts']
A_dim_counts = data['A_dimer_counts']
A_trim_counts = data['A_trimer_counts']

B_mon_counts = data['B_mon_counts']
B_dim_counts = data['B_dimer_counts']
B_trim_counts = data['B_trimer_counts']

C_mon_counts = data['C_mon_counts']
C_dim_counts = data['C_dimer_counts']
C_trim_counts = data['C_trimer_counts']

A_count = jnp.concatenate([A_mon_counts, A_dim_counts, A_trim_counts])
B_count = jnp.concatenate([B_mon_counts, B_dim_counts, B_trim_counts])
C_count = jnp.concatenate([C_mon_counts, C_dim_counts, C_trim_counts])
nper_structure = jnp.array([A_count, B_count, C_count])

conc_A = 19.
conc_B = 19.
conc_C = 18.

init_conc = jnp.array([conc_A, conc_B, conc_C])

def create_optimizer(lr, clip_norm):
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr)
    )

def ofer(conc):
    m_conc = conc
    tot_conc = jnp.sum(m_conc)
    log_mon_conc = safe_log(m_conc)
    
    def loss_fn(log_concs_struc):
        def mon_loss_fn(mon_idx):
            mon_val = jnp.log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
            return mon_val - log_mon_conc[mon_idx]

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

            return log_vcs - vcs_denom - log_zs + z_denom
        
        mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
        struc_loss = vmap(struc_loss_fn)(jnp.arange(n, tot_num_structures))
        combined_loss = jnp.concatenate([mon_loss, struc_loss])
        loss_std = jnp.std(combined_loss)

        tot_loss = jnp.linalg.norm(combined_loss) + loss_std
        return tot_loss, combined_loss
        #tot_loss = jnp.linalg.norm(combined_loss)
        #return tot_loss

    struc_concs_guess = jnp.full(tot_num_structures, safe_log(tot_conc / tot_num_structures))

    #optimizer = create_optimizer(2e-2, 1.)
    optimizer = optax.adam(2e-2)
    params = struc_concs_guess
    opt_state = optimizer.init(params)
    grad_fn = jit(value_and_grad(loss_fn, has_aux=True))
    
    loss_threshold = .1
    max_inner_iters = 1000
    max_sets = 500

    losses = []
    for _ in range(max_sets):
        for _ in range(max_inner_iters):
            (loss_val, combined_loss), grads = grad_fn(params)
            losses.append(loss_val)
            if loss_val < loss_threshold:
                break
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        else:
            continue 
        break 

        
    fin_log_concs = deepcopy(params)
    fin_concs = jnp.exp(fin_log_concs)

    yields = fin_concs / jnp.sum(fin_concs)
    
    target_yield = safe_log(yields[-1])
    offtarget_dim = safe_log(yields[5])
    offtarget_tri = safe_log(yields[-4])
    yield_sum = safe_log(jnp.sum(yields))
    monomerA = safe_log(yields[0])
    monomerB = safe_log(yields[1])
    monomerC = safe_log(yields[2])
    return target_yield, offtarget_dim , offtarget_tri, monomerA,  monomerB, monomerC, yield_sum

def ofer_grad_fn(conc):
    target_yield = ofer(conc)[0]
    return -target_yield

def project(param):
    return jnp.clip(param, a_min=1e-4)

our_grad_fn = value_and_grad(ofer_grad_fn)

outer_optimizer = create_optimizer(1e-1, 1.)
params = init_conc
opt_state = outer_optimizer.init(params)

n_outer_iters = 200
outer_losses = []

with open("optimization_log_tri.txt", "w") as log_file:
    log_file.write("Iteration\tTarget_Yield\tOfftarget_Dim\tOfftarget_Tri\tMonA\tMonB\tMonC\tTot_yield\n")

    for i in tqdm(range(n_outer_iters)):
        loss_val, grads = our_grad_fn(params)
        outer_losses.append(loss_val)
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = project(params)
        
        target_yield, offtarget_dim, offtarget_tri, monA, monB, monC, tot_yield = ofer(params)
        finite_diff_grads = finite_difference_grad(ofer_grad_fn, params)

        log_file.write(f"{i+1}\t{target_yield}\t{offtarget_dim}\t{offtarget_tri}\t{monA}\t{monB}\t{monC}\t{tot_yield}\n")
        
        print(f"Iteration {i+1}/{n_outer_iters}")
        print(f"Loss: {loss_val}")
        print(f"Yield: {jnp.exp(target_yield)}")
        print(f"Param: {params}")
        print(f"Gradients: {grads}")
        print(f"Gradients (Finite Difference): {finite_diff_grads}")


final_param = params
final_target_yield = ofer(final_param)[0]
print(f"Final Optimized Parameter (monomer concentration): {final_param}")
print(f"Final Target Yield: {final_target_yield}")


