
from utils import cumsum

class FixedItAnalysis:
    def __init__(self, vs, H, diag_shift, distance):
        self.vs = vs
        self.H = H

def compute_new_dp_norm_psi(j, pdf, Hloc, jac_orig, diag_shift, im_t_ev, vs, delta=0.0001):
    # set sample prob to 0
    pdf_new = pdf.at[j].set(0)
    # renormalize pdf
    pdf_new = pdf_new/(jnp.sum(pdf_new))

    # compute new jacobian: recenter and sqrt rescale
    jac_new = jnp.sqrt(pdf_new[:,None])*(jac_orig - jnp.sum(jac_orig*pdf_new[:,None],axis=0))
    # center and scale H_loc too
    Hloc_new = jnp.sqrt(pdf_new)*(Hloc - jnp.sum(Hloc*pdf_new))

    # recompute qgt and rhs
    new_rhs = jac_new.transpose().conj() @ Hloc_new
   
    new_qgt = (jac_new.transpose().conj() @ jac_new) + 1.0e-10*jnp.eye(jac_c.shape[1]) 
    # new_qgt = 1.0e-4*jnp.eye(jac_c.shape[1]) 
    # Solve system
    dp_sol =  nk.optimizer.solver.cholesky(new_qgt, new_rhs)[0]
    # update params and compute update vstate
    params , tree_def= jax.tree_util.tree_flatten(vs.parameters)
    leaf_sizes = [leaf.size for leaf in params]
    partitioned_dp = jnp.split(dp_sol, cumsum(leaf_sizes)[:-1])
    new_leaves = [params[i] - delta*partitioned_dp[i].reshape(params[i].shape) for i in range(len(params))]
    new_pars = jax.tree_util.tree_unflatten(tree_def, new_leaves)
    log_psi_updated = vs.model.apply({"params": new_pars},vs.hilbert.all_states())
    psi_new = jnp.exp(log_psi_updated)
    # print(fid_metric(psi_norm, im_t_ev))
    return fid_metric(psi_new, im_t_ev).real

delta = 0.0001
im_t_ev = vs.to_array() - delta* H @ vs.to_array()

fid_vals = nk.jax.vmap_chunked(lambda j: compute_new_dp_norm_psi(j, pdf, Hloc, jacobian_orig, diag_shift, im_t_ev, vs, delta = delta), chunk_size=None)(jnp.arange(jacobian.shape[0]))
in_vals = jnp.argsort(fid_vals)



infid_ev_init = []
jac_ev = jacobian_orig
pdf_r = pdf
# print(pdf_r)
for idx, basis_id in enumerate(in_idx):
    # remove the sample
    pdf_r = pdf_r.at[basis_id].set(0)
    # renormalize pdf
    S = jnp.sum(pdf_r)
    if S >0:
        pdf_r = pdf_r/S
    # if idx == 510:
    #     print(pdf_r)
    # compute new jacobian: recenter and sqrt rescale
    jac_ev = jnp.sqrt(pdf_r[:,None])*(jacobian_orig - jnp.sum(jacobian_orig*pdf_r[:,None],axis=0))

    h_loc_new = jnp.sqrt(pdf_r)*(Hloc - jnp.sum(Hloc*pdf_r))
    new_rhs = jac_ev.T.conj() @ h_loc_new
    # rhs_overlap.append(fid_metric(force_fs, rhs))
    # # recompute qgt and rhs
    new_qgt = (jac_ev.transpose().conj() @ jac_ev) + diag_shift*jnp.eye(jac_ev.shape[1])
    # Solve system
    dp_sol =  nk.optimizer.solver.cholesky(new_qgt, new_rhs)[0]
    # update params and compute update vstate
    params , tree_def= jax.tree_util.tree_flatten(vs.parameters)
    leaf_sizes = [leaf.size for leaf in params]
    partitioned_dp = jnp.split(dp_sol, cumsum(leaf_sizes)[:-1])
    new_leaves = [params[i] - delta*partitioned_dp[i].reshape(params[i].shape) for i in range(len(params))]
    new_pars = jax.tree_util.tree_unflatten(tree_def, new_leaves)
    log_psi_updated = vs.model.apply({"params": new_pars},vs.hilbert.all_states())
    psi_norm = jnp.exp(log_psi_updated)
    # print(fid_metric(psi_norm, im_t_ev))
    infid_ev_init.append(fid_metric(psi_norm, im_t_ev).real)
    # eigv, eigV = jnp.linalg.eigh(qgt)
    # if eigv[-1]/eigv[0] < 0:
    #     print(eigv)
    # qgt_cond_num.append(eigv[-1]/eigv[0])
    # qgt_rank.append(jnp.sum(eigv > 10e-5))
# print(pdf_r)