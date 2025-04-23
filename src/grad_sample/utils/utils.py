import netket.jax as nkjax
import os
from scipy.sparse.linalg import eigsh
from netket.vqs import FullSumState
import jax.numpy as jnp
from grad_sample.is_hpsi.expect import snr_comp
import copy
import jax
from advanced_drivers.driver import statistics
from netket_checkpoint._src.serializers.metropolis import serialize_MetropolisSamplerState, deserialize_MetropolisSamplerState
from flax.serialization import msgpack_serialize

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

def save_alpha(step, logdata, driver):
    if step%20 ==0:
        print(driver._ham.is_mode)
        logdata['alpha'] = driver._ham.is_mode
    return True

def save_rel_err_fs(step, logdata, driver, fs_state, e_gs, save_every=1, output_dir=None):
    if driver.step_count % save_every == 0:
        fs_state.variables = copy.deepcopy(driver.state.variables)
        # e = fs_state.expect(driver._ham.operator).mean.real
        try:
            # is operator case
            e = fs_state.expect(driver._ham.operator).mean.real
        except: 
            e = fs_state.expect(driver._ham).mean.real

        logdata["rel_err"] = jnp.abs(e-e_gs)/jnp.abs(e_gs)
        # if output_dir!=None and driver.step_count % 10*save_every == 0:
        #     fig, ax = plt.subplots()
        #     e_r_fs = logdata["rel_err"]
        #     ax.plot(e_r_fs["iters"], e_r_fs["value"], label= "FullSum")
            
        #     ax.set_title(f"Partial training curve")
        #     ax.set_xlabel("iteration")
        #     ax.set_ylabel("Relative error")
        #     ax.set_yscale("log")
        #     plt.savefig(output_dir + '/training_partial.png')
        #     plt.clf()
    return True

def save_rel_err_large(step, logdata, driver, e_ref, n_s = 2**13, n_sites=None, save_every=1, output_dir=None):
    # compare an estimate of the energy with a large number of samples to a litterature reference
    n_s_orig = driver.state.n_samples
    driver.state.n_samples = n_s
    if driver.step_count % save_every == 0:
        g, l, w = driver.local_estimators()
        energy = statistics(l,w)
        e = energy.mean.real
        logdata['MC2_mean'] = e
        logdata['MC2_var'] = energy.variance
        logdata['MC2_err'] = energy.error_of_mean
        logdata['MC2_R_hat'] = energy.R_hat
        logdata['MC2_tau'] = energy.tau_corr
        
        if n_sites != None:
            logdata["rel_err"] = (e/4/n_sites -e_ref)/jnp.abs(e_ref)
        else:
            logdata["abs_err"] = jnp.abs(e-e_ref) #absolute error for qchem
            logdata['rel_err'] = jnp.abs(e-e_ref)/jnp.abs(e_ref)
        # if output_dir!=None and driver.step_count % 2*save_every == 0:
        #     fig, ax = plt.subplots()
        #     e_r_fs = logdata["rel_err"]
        #     ax.plot(e_r_fs["iters"], e_r_fs["value"], label= "FullSum")
            
        #     ax.set_title(f"Partial training curve")
        #     ax.set_xlabel("iteration")
        #     ax.set_ylabel("Relative error")
        #     ax.set_yscale("log")
        #     plt.savefig(output_dir + '/training_partial.png')
        #     plt.clf()
    driver.state.n_samples = n_s_orig
    return True

def save_sampler_state(step, logdata, driver, out_prefix, save_every=50):
    if step % save_every ==0:
        for chain_name, sampler_state in driver.state.sampler_states.items():
            state_dict = serialize_MetropolisSamplerState(sampler_state)
            binary_data = msgpack_serialize(state_dict)
            with open(os.path.join(out_prefix, "sampler_state_%s.mpack"%chain_name), "wb") as outfile:
                outfile.write(binary_data)
    return True

def save_snr(step, logdata, driver, save_every=1):
    if driver.step_count % save_every == 0:
        snr_jac, snr_f = snr_comp(driver.state, driver._ham, chunk_size=driver.state.chunk_size)
        logdata['snr_f'] = snr_f
        logdata['snr_jac'] = snr_jac
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

def find_closest_saved_vals(E_err, saved_vals, save_every, n_vals_per_scale=1):
    L = len(E_err)
    exp_max = int(jnp.log10(jnp.max(E_err))) +1
    exp_min = int(jnp.log10(jnp.min(E_err))) -1
    exp_list = jnp.flip(jnp.arange(exp_min, exp_max))
    print((exp_max - exp_min)*n_vals_per_scale)
    exp_list = jnp.flip(jnp.linspace(exp_min, exp_max, (exp_max - exp_min +1)*n_vals_per_scale))
    print(exp_list)
    target_values = 10.0 ** (exp_list)  # 10^n values
    print(target_values)
    closest_saved_vals = []
    error_val = []
    for target in target_values:
        # Find the index in E_err with the value closest to the target
        closest_index = jnp.abs(E_err - target).argmin()

        # Find the corresponding index in saved_vals
        # `saved_vals` is of length L//10 and corresponds to every 10th iteration
        saved_index = closest_index // save_every
        saved_index = min(saved_index, len(saved_vals) - 1)  # Ensure index is within bounds
        if saved_vals[saved_index] not in closest_saved_vals:
            closest_saved_vals.append(saved_vals[saved_index])
            error_val.append(E_err[saved_index*save_every])
    return closest_saved_vals, error_val

def compute_snr_callback(step, logdata, driver, fs_state:FullSumState, H_sp, save_every=10, chunk_size_jac=200):
    # estimate local grad
    if step % save_every == 0:
        # fs_state = FullSumState(hilbert = driver.state.hilbert, model = driver.state.model, chunk_size=None, seed=0)
        fs_state.variables = copy.deepcopy(driver.state.variables)
        pdf = fs_state.probability_distribution()
        vstate_arr = fs_state.to_array()
        Hloc = H_sp @ vstate_arr / vstate_arr
        Hloc_c = (Hloc - jnp.sum(Hloc*pdf))
        mode = "complex"
        # uncentered jacobian
        jacobian_orig = nkjax.jacobian(
            fs_state._apply_fun,
            fs_state.parameters,
            fs_state.hilbert.all_states(), #in MC state, this is vstate.samples
            fs_state.model_state,
            pdf=pdf,
            mode=mode,
            dense=True,
            center=False,
            chunk_size=200,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )

        # (#ns, 2) -> (#ns*2)
        Hloc_2 = jnp.stack([jnp.real(Hloc_c), jnp.imag(Hloc_c)], axis=-1)
        Hloc_c = jax.lax.collapse(Hloc_2, 0, 2)
        jacobian_orig_c = jacobian_orig - jnp.sum(jacobian_orig*jnp.expand_dims(pdf, range(len(jacobian_orig.shape))[1:]),axis=0)
        jacobian_orig_c = jax.lax.collapse(jacobian_orig_c, 0, 2)
        loc_grad_v = jacobian_orig_c.T * Hloc_c
        loc_grad_v = loc_grad_v[:, ::2] + loc_grad_v[:, 1::2]
        
        # print(loc_grad_v.shape)
        # n_p = loc_grad_v.shape[0]//2
        # print(loc_grad_v_holo - (loc_grad_v[:n_p,:] + 1j* loc_grad_v[n_p:,:]))

        mean_grad_unc = jnp.sum(jnp.abs(pdf * loc_grad_v), axis=0) / jnp.sum(jnp.abs(pdf * loc_grad_v))
        
        def unnorm_pdf(alpha):
            return (jnp.abs(vstate_arr)**alpha)
        
        def compute_snr(q):
            q_pdf = q / jnp.sum(q)
            w_mean = jnp.sum(q_pdf * unnorm_pdf(2.0)/q)**2
            v = jnp.sum(q_pdf * (unnorm_pdf(2.0)/q)**2 * jnp.abs(loc_grad_v - jnp.sum(pdf * loc_grad_v, axis=1)[:, None])**2, axis=1)/w_mean
            return jnp.mean(jnp.abs(jnp.sum(pdf * loc_grad_v, axis = 1)) / jnp.sqrt(v))

        a_vals = jnp.linspace(0.01, 2, 200)
        snr_a = jnp.array([compute_snr(unnorm_pdf(a)) for a in a_vals])
        
        # Initialize a dummy carry (not used, but required)
        init_carry = 0.0  

        argmax_index = jnp.argmax(snr_a)
        argmax_a = a_vals[argmax_index]
        # print(jnp.mean(unnorm_pdf(2.0) * jnp.abs(loc_grad_v), axis=0).shape)
        snr_grad = compute_snr(jnp.mean(unnorm_pdf(2.0) * jnp.abs(loc_grad_v), axis=0))
        logdata['snr_a'] = snr_a[::20]
        logdata['snr_grad'] = snr_grad
        logdata['max_snr_a'] = jnp.max(snr_a)
        logdata['argmax_snr_a'] = argmax_a
        logdata['snr_psi_sq'] = compute_snr(unnorm_pdf(2.0))
        # logdata['snr_hpsi'] = compute_snr(jnp.abs(driver._ham @ vstate_arr))
        # logdata['grad_mag'] = jnp.mean(jnp.abs(loc_grad_v), axis=0)
    return True
        