from basic_test import RBM, _setup
import jax
import jax.numpy as jnp
from tqdm import tqdm
import netket as nk
# import netket_pro as nkp
import netket.jax as nkjax
import flax
import matplotlib.pyplot as plt
import os
import flax.serialization
import copy

import argparse
import yaml
# from utils import load_yaml_to_vars
from utils import cumsum

def load_yaml_to_vars(yaml_path):
    # Open and parse the YAML file
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create variables dynamically
    for key, value in config.items():
        globals()[key] = value
        print(f"{key} = {globals()[key]}")  # Print each variable for confirmation

def compute_new_dp_norm(j, pdf, Hloc, jac_orig, diag_shift, exact_dp):
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
    # almost same val
    # print(new_rhs / (jac_new.transpose().conj() @ (jnp.sqrt(pdf)*(Hloc_new-jnp.sum(Hloc_new*pdf)))))
    new_qgt = (jac_new.transpose().conj() @ jac_new) + diag_shift*jnp.eye(jac_new.shape[1])  
    # Solve system
    dp_sol =  nk.optimizer.solver.cholesky(new_qgt, new_rhs)[0]
    
    return fid_metric(exact_dp, dp_sol).real

def fid_metric(dp_exact, dp_approx):
    dot = jnp.dot(dp_approx.conj(),dp_exact)*jnp.dot(dp_approx, dp_exact.conj())
    norm = (jnp.dot(dp_approx, dp_approx.conj())*jnp.dot(dp_exact.conj(), dp_exact))
    return dot/norm

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
    # almost same val
    # print(new_rhs / (jac_new.transpose().conj() @ (jnp.sqrt(pdf)*(Hloc_new-jnp.sum(Hloc_new*pdf)))))
    new_qgt = (jac_new.transpose().conj() @ jac_new) + diag_shift*jnp.eye(jac_new.shape[1])  
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
    return fid_metric(psi_norm, im_t_ev).real

def main():
    
    parser = argparse.ArgumentParser(description="Parse YAML config to variables.")
    parser.add_argument("-y", "--y", required=False, type=str, help="Path to the YAML configuration file.", default="config/base_config.yaml")

    # Parse arguments
    args = vars(parser.parse_args())
    load_yaml_to_vars(args["y"])
    print(globals())
    os.environ["CUDA_VISIBLE_DEVICES"]= device
    # L = 3
    # full_summation = True
    # lr = 0.01
    # n_samples = 512
    
    # holomorphic = True
    # alpha = 1
    # diag_shift = 1e-4
    
    # holomorphic = True
    # solver_fn = nk.optimizer.solver.cholesky
    # sr = nk.optimizer.SR(solver=solver_fn, diag_shift=diag_shift, holomorphic=holomorphic)

    save_every = 3
    output_prefix = f"/scratch/.amisery/optimization_run/state_{save_every}_{alpha}_{L}_{int(-jnp.log10(diag_shift))}"

    is_complex=True

    H, opt, vstate = _setup(
        L=L,
        n_samples=n_samples,
        lr=lr,
        is_complex=is_complex,
        full_summation=full_summation,
        alpha=alpha
    )

    eval_s = jnp.concatenate([jnp.floor(jnp.linspace(6//3, 54//3, 8)).astype(int), jnp.array([j**1.8 for j in range(5,11)]).astype(int)])
    # eval_s = jnp.array([j**1.8 for j in range(5,11)]).astype(int)
    print(eval_s)

    
    for it in eval_s:
        # load pars/vstate from mpack file
        print(it)
        use_state = True
        variables = None
        vs = copy.copy(vstate)
        with open(output_prefix + "/%d.mpack"%it, 'rb') as file:
        # cannot use from bytes directly
            vs.variables = flax.serialization.from_bytes(variables, file.read())

        # Compute jacobian in full summation using the new vstate
        pdf = vs.probability_distribution()
        H_sp = H.to_sparse()
        # to array returns the exponential of vs.model.apply (log wf), normalized
        Hloc = H_sp @ vs.to_array() / vs.to_array()

        # Hloc = Hloc - jnp.mean(Hloc*pdf) Hloc not centered bc it's handled in the computation
        Hloc_c = jnp.sqrt(pdf)*(Hloc - jnp.sum(Hloc*pdf))
        # Hloc = compute_eloc_vstate(vs, H, vs.hilbert.all_states()) #always
        chunk_size_jac = 2
        chunk_size_vmap = 1
        # jacobian not centered bc it is handled dynamically in the computation
        jacobian_orig = nkjax.jacobian(#only for FullSumState
            vs._apply_fun,
            vs.parameters,
            vs.hilbert.all_states(), #in MC state, this is vstate.samples
            vs.model_state,
            pdf=pdf,
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=chunk_size_jac,
            _sqrt_rescale=False, #rescale by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
        delta = 0.0001
        psi = vs.to_array()
        im_t_ev = psi - delta*H@psi
        # Compute jacobian in full summation using the new vstate
        jac_c = jnp.sqrt(pdf[:,None])*(jacobian_orig - jnp.sum(jacobian_orig*pdf[:,None],axis=0))

        force_fs = jac_c.transpose().conj() @ Hloc_c
        S_fs = jac_c.transpose().conj() @ jac_c + diag_shift*jnp.eye(jac_c.shape[1])

        exact_dp = nk.optimizer.solver.cholesky(S_fs, force_fs)[0]

        sim_tol = 0.9
        sim = 1
        base_idx = jnp.arange(jacobian_orig.shape[0])
        
        removed_indices = []
        fid_ev = []
        pdf_r = pdf
        
        def get_new_sim(pdf, base_idx):
            if use_state:
                fid_vals = nk.jax.vmap_chunked(lambda j: compute_new_dp_norm_psi(j, pdf, Hloc, jacobian_orig, diag_shift, im_t_ev=im_t_ev, vs=vs, delta=delta ), chunk_size=chunk_size_vmap)(base_idx)
            else:
                fid_vals = nk.jax.vmap_chunked(lambda j: compute_new_dp_norm(j, pdf, Hloc, jacobian_orig, diag_shift, exact_dp), chunk_size=chunk_size_vmap)(base_idx)
            
            if len(fid_vals) > 0:
                j_el = jnp.argmax(fid_vals)
                basis_id = base_idx[j_el]
            else : 
                return -1, -1, -1
            return fid_vals[j_el], basis_id, j_el
        
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        progress_bar = tqdm()
        while sim > sim_tol:
            sim, basis_id, j_el = jax.jit(get_new_sim)(pdf_r, base_idx)
            if sim == -1:
                # we've removed all samples
                break
            # set to 0 the least important sample for accurate NGD
            pdf_r = pdf_r.at[j_el].set(0)
            # renormalize pdf
            pdf_r = pdf_r/jnp.sum(pdf_r)

            fid_ev.append(sim)
            removed_indices.append(basis_id)
            base_idx = jnp.delete(base_idx, j_el)
            progress_bar.update(1)

        progress_bar.close()

        removed_indices = jnp.array(removed_indices)
        fid_ev = jnp.array(fid_ev)
        if use_state:
            jnp.savez(output_prefix + "/fid_ev%d_state.npz"%it, fid_ev)
            jnp.savez(output_prefix + "/removed_indices%d_state.npz"%it, removed_indices)
        else:    
            jnp.savez(output_prefix + "/fid_ev%d.npz"%it, fid_ev)
            jnp.savez(output_prefix + "/removed_indices%d.npz"%it, removed_indices)
        print("files saved")
    
if __name__=="__main__":
    main()