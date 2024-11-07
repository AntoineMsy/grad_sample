import netket.jax as nkjax
from scipy.sparse.linalg import eigsh

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

def e_diag(H_sp):
    eig_vals, eig_vecs = eigsh(
        H_sp, k=2, which="SA"
    )  # k is the number of eigenvalues desired,
    E_gs = eig_vals[0]  # "SA" selects the ones with smallest absolute value
    return E_gs