import netket.jax as nkjax
from scipy.sparse.linalg import eigsh
from netket.vqs import FullSumState
import copy
def cumsum(lst):
    cumulative_sum = []
    total = 0
    for num in lst:
        total += num
        cumulative_sum.append(total)
    return cumulative_sum

def save_cb(step, logdata, driver):
    dp = driver._dp
    dp, _ = nkjax.tree_ravel(dp)
    logdata["dp"] = dp
    return True

def save_rel_err_fs(step, logdata, driver, fs_state, e_gs, save_every=1):
    if driver.step_count % save_every == 0:
        fs_state.variables = copy.deepcopy(driver.state.variables)
        try:
            # is operator case
            e = fs_state.expect(driver._ham.operator).mean.real
        except: 
            e = fs_state.expect(driver._ham).mean.real

        logdata["rel_err"] = jnp.abs(e-e_gs)/jnp.abs(e_gs)
    return True

def save_rel_err(step, logdata, driver, e_gs, save_every=1):
    e = driver.energy.mean
    logdata["rel_err"] = jnp.abs(e-e_gs)/jnp.abs(e_gs)
    return True
    
def e_diag(H_sp):
    eig_vals, eig_vecs = eigsh(
        H_sp, k=2, which="SA"
    )  # k is the number of eigenvalues desired,
    E_gs = eig_vals[0]  # "SA" selects the ones with smallest absolute value
    return E_gs

import jax.numpy as jnp

def find_closest_saved_vals(E_err, saved_vals, save_every):
    L = len(E_err)
    exp_max = int(jnp.log10(jnp.max(E_err))) +1
    exp_min = int(jnp.log10(jnp.min(E_err))) -1
    exp_list = jnp.flip(jnp.arange(exp_min, exp_max))
    
    target_values = 10.0 ** (exp_list)  # 10^n values
    print(target_values)
    closest_saved_vals = []

    for target in target_values:
        # Find the index in E_err with the value closest to the target
        closest_index = jnp.abs(E_err - target).argmin()

        # Find the corresponding index in saved_vals
        # `saved_vals` is of length L//10 and corresponds to every 10th iteration
        saved_index = closest_index // save_every
        saved_index = min(saved_index, len(saved_vals) - 1)  # Ensure index is within bounds

        closest_saved_vals.append(saved_vals[saved_index])

    return closest_saved_vals

