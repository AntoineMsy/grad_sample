import jax
import jax.numpy as jnp

import netket as nk
# import netket_pro as nkp
import netket.jax as nkjax
import netket.stats as nkstats
import optax
import flax.linen as nn
import os
import argparse
import yaml
from post_proc_analysis import compute_new_dp_norm_psi, fid_metric
from utils import cumsum

from scipy.sparse.linalg import eigsh
from tqdm import tqdm

class RBM(nn.Module):
    num_hidden: int  # Number of hidden neurons
    is_complex: bool
    real_output: bool = False

    def setup(self):
        self.linearR = nn.Dense(
            features=self.num_hidden,
            use_bias=True,
            param_dtype=jnp.float64,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.normal(stddev=0.02),
        )
        if self.is_complex:
            self.linearI = nn.Dense(
                features=self.num_hidden,
                use_bias=False,
                param_dtype=jnp.float64,
                kernel_init=jax.nn.initializers.normal(stddev=0.02),
                bias_init=jax.nn.initializers.normal(stddev=0.02),
            )

    def __call__(self, x):
        x = self.linearR(x)

        if self.is_complex:
            x = x + 1j * self.linearI(x)

        x = jnp.log(jax.lax.cosh(x))

        if self.real_output:
            return jnp.sum(x, axis=-1)
        elif self.is_complex:
            return jnp.sum(x, axis=-1)
        else:
            return jnp.sum(x, axis=-1).astype(jnp.complex128)



def _setup(
    *,
    L=3,
    n_samples=512,
    n_discard_per_chain=0,
    lr=0.01,
    is_complex=True,
    machine=None,
    real_output=False,
    chunk_size=None,
    full_summation=False,
    alpha = 1
):
    Ns = L * L
    # lattice = nk.graph.Square(L, max_neighbor_order=2)
    lattice = nk.graph.Square(L, pbc=True)
    
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0)
    if nk.config.netket_experimental_sharding:
        H = H.to_jax_operator()

    # Define a variational state
    if machine is None:
        # machine = nk.models.RBM()
        # machine = RBM(num_hidden=Ns, is_complex=is_complex, real_output=real_output)
        
        machine = nk.models.RBM(alpha=alpha, param_dtype=complex)

    opt = optax.sgd(learning_rate=lr)
    
    if full_summation:
        vstate = nk.vqs.FullSumState(hilbert=hi, model=machine, chunk_size=chunk_size, seed=0)
    else:    
        sampler = nk.sampler.ExactSampler(hilbert=hi,)
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=machine,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            seed=0,
            sampler_seed=0,
            chunk_size=chunk_size,
        )

    return H, opt, vstate

def save_cb(step, logdata, driver):
    dp = driver._dp
    dp, _ = nkjax.tree_ravel(dp)
    logdata["dp"] = dp
    return True

def load_yaml_to_vars(yaml_path):
    # Open and parse the YAML file
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Create variables dynamically
    for key, value in config.items():
        globals()[key] = value
        print(f"{key} = {globals()[key]}")  # Print each variable for confirmation

def setup_and_run(fullsum = True, fid_stride = 10, num_removed = 200):
    # fullsum: full sum if true or gradually remove 50,100,150,200 samples to compute the variations in update fid, and choose the best value (for now fixed to 100)
    
    rel_err_l = []
    fid_diff_l = []

    H, opt, vstate = _setup(
        L=L,
        n_samples=n_samples,
        lr=lr,
        is_complex=is_complex,
        full_summation=full_summation, 
        alpha = alpha
    )
    # print exact diag energy
   

    H_sp = H.to_sparse()
        
    eig_vals, eig_vecs = eigsh(
        H_sp, k=2, which="SA"
    )  # k is the number of eigenvalues desired,
    E_gs = eig_vals[0]  # "SA" selects the ones with smallest absolute value

    print("The ground state energy is:", E_gs)
    json_log = nk.logging.JsonLog(output_prefix=output_prefix)
    state_log = nk.logging.StateLog(output_prefix=output_prefix, save_every=save_every)
    gs = nk.VMC(hamiltonian=H, optimizer=opt, variational_state=vstate, preconditioner=sr)
    # gs.run(n_iter=n_iter, out=(json_log, state_log), callback=(save_cb,))
    delta = lr/10 #defined to compute infidielity changes
    #at every step find the least num_removed samples to removed by computing the fidelity
    # rewrite training loop with fid computation

    for k in tqdm(range(n_iter)):
        # Compute jacobian in full summation using the new vstate
        pdf = vstate.probability_distribution()
        H_sp = H.to_sparse()
        # to array returns the exponential of vs.model.apply (log wf), normalized
        Hloc = H_sp @ vstate.to_array() / vstate.to_array()

        Hloc_c = jnp.sqrt(pdf)*(Hloc - jnp.sum(Hloc*pdf))

        chunk_size_jac = 2
        chunk_size_vmap = 1
        # jacobian not centered bc it is handled dynamically in the computation
        jacobian_orig = nkjax.jacobian(#only for FullSumState
            vstate._apply_fun,
            vstate.parameters,
            vstate.hilbert.all_states(), #in MC state, this is vstate.samples
            vstate.model_state,
            pdf=pdf,
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=chunk_size_jac,
            _sqrt_rescale=False, #rescale by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
        jac_c = jacobian_orig - jnp.sum(jacobian_orig*pdf[:,None],axis=0)

        jac_c = jnp.sqrt(pdf[:,jnp.newaxis])*jac_c
        if k%fid_stride == 0: # compute removed state every 10 its
            im_t_ev = vstate.to_array() - delta* H @ vstate.to_array()

            fid_vals = jax.jit(nk.jax.vmap_chunked(lambda j: compute_new_dp_norm_psi(j, pdf, Hloc, jacobian_orig, diag_shift, im_t_ev, vstate, delta = delta), chunk_size=None))(jnp.arange(jacobian_orig.shape[0]))
            jac_ev = jacobian_orig
            in_vals = jnp.argsort(1 - fid_vals)

        # take 100 first as a principle
        if fullsum:
            pdf_r = pdf
        else:
            pdf_r = pdf.at[in_vals[:num_removed]].set(0)
            # renormalize pdf
            S = jnp.sum(pdf_r)
            if S >0:
                pdf_r = pdf_r/S
        
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
        params , tree_def= jax.tree_util.tree_flatten(vstate.parameters)
        leaf_sizes = [leaf.size for leaf in params]
        partitioned_dp = jnp.split(dp_sol, cumsum(leaf_sizes)[:-1])
        new_leaves = [params[i] - lr*partitioned_dp[i].reshape(params[i].shape) for i in range(len(params))]
        new_pars = jax.tree_util.tree_unflatten(tree_def, new_leaves)
        psi_trunc = jnp.exp(vstate.model.apply({"params": new_pars}, vstate.hilbert.all_states()))

        if not fullsum:
            # also compute orig infid to compare 
            force_fs = jac_c.transpose().conj() @ Hloc_c 
            S_fs = (jac_c.transpose().conj() @ jac_c) + 1.0e-10*jnp.eye(jac_c.shape[1])

            exact_dp = nk.optimizer.solver.cholesky(new_qgt, new_rhs)[0]
            params , tree_def= jax.tree_util.tree_flatten(vstate.parameters)
            leaf_sizes = [leaf.size for leaf in params]
            partitioned_dp = jnp.split(exact_dp, cumsum(leaf_sizes)[:-1])
            leaves_full = [params[i] - lr*partitioned_dp[i].reshape(params[i].shape) for i in range(len(params))]
            pars_full = jax.tree_util.tree_unflatten(tree_def, new_leaves)
            psi_full = jnp.exp(vstate.model.apply({"params": pars_full}, vstate.hilbert.all_states()))

            # compare fidelities
            fid_diff_l.append(jnp.abs(fid_metric(psi_trunc, im_t_ev).real - fid_metric(psi_full, im_t_ev).real))

        # set vstate params to new_pars
        vstate.parameters = new_pars
        rel_err = jnp.abs(vstate.expect(H).mean.real - E_gs)/jnp.abs(E_gs)
        rel_err_l.append(rel_err)
        if k%20 == 0:
            # print relative error to gs
            print(rel_err)
        # log_psi_updated = vs.model.apply({"params": new_pars},vs.hilbert.all_states())
        # psi_norm = jnp.exp(log_psi_updated)

    if fullsum:
        return rel_err_l
    else:
        return rel_err_l, fid_diff_l
    
    
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
    n_iter = 1000
    # alpha = 2
    # diag_shift = 1e-4
    # holomorphic = True
    solver_fn = nk.optimizer.solver.cholesky
    sr = nk.optimizer.SR(solver=solver_fn, diag_shift=diag_shift, holomorphic=holomorphic)
    
    save_every = 3
    output_prefix = f"/scratch/.amisery/optimization_run/state_{save_every}_{alpha}_{L}_{int(-jnp.log10(diag_shift))}"
    
    is_complex=True
    real_output=False
    

    rel_err_full = setup_and_run(fullsum=True)

    rel_err_trunc, infid_ev = setup_and_run(fullsum=False)

    out_dict = {"err_full" : rel_err_full, "err_trunc" : rel_err_trunc, "infid_ev": infid_ev}
    jnp.savez("removal_training.npz", out_dict)
    
"""
To compute local energies:
    Hloc = vstate.local_energies(H) #always
    
    jacobians = nkjax.jacobian( #only for FullSumState
        vstate._apply_fun,
        vstate.parameters,
        vstate_all_states(), #in MC state, this is vstate.samples
        vstate.model_state,
        pdf=vstate.probability_distribution(), #in MC state, this is None
        mode="complex",
        dense=True,
        center=True,
        chunk_size=chunk_size,
        _sqrt_rescale=True, #rescale by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
    )
"""