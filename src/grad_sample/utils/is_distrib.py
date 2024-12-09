import jax.numpy as jnp
import jax
from grad_sample.utils.misc import compute_eloc
import netket.jax as nkjax

def modulus(apply_fun, H, params, sigma):
    return jnp.log(jnp.sqrt(jnp.abs(jnp.exp(apply_fun(params, sigma)))))

def Hpsi(apply_fun, H, params, sigma):
    eta, eta_H_sigma = H.get_conn_padded(sigma)
    log_Hpsi = jnp.log(jnp.sum(eta_H_sigma  * jnp.exp(apply_fun(params, eta)),axis=-1))
    return log_Hpsi

def Hlocpsi(apply_fun, H, params, sigma):
    # currently |Hloc(\sigma)\psi(\sigma)|^2
    Hloc = compute_eloc(apply_fun, params, H, sigma)
    # log_Hloc_c = jnp.log((1/jnp.sqrt(len(sigma)))*(Hloc - jnp.mean(Hloc)))
    # return log_Hloc_c + apply_fun(params, sigma)
    return jnp.log(jnp.sqrt(jnp.abs(Hloc)))+ apply_fun(params, sigma)



def grad_psi(apply_fun, H, params, sigma):
    psi = apply_fun(params, sigma)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    grad = jnp.abs(jnp.sum(jac, axis=1))
    return jnp.log(grad) + psi

def grad_psi_delta(apply_fun, H, params, sigma, jac_mean):
    psi = apply_fun(params, sigma)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    grad = jnp.abs(jnp.sum(jac -jac_mean, axis=1))
    return jnp.log(grad) + psi

def hloc_grad_psi(apply_fun, H, params, sigma):
    hloc = compute_eloc(apply_fun, params, H, sigma)
    psi = apply_fun(params, sigma)
    # _, grad = jax.jvp(lambda pars: apply_fun(pars, sigma), [params], [params])
    # print(grad.shape)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    grad = jnp.abs(jnp.sum(jac, axis=1))
    return jnp.log(jnp.sqrt(grad)) + psi + jnp.log(jnp.sqrt(jnp.abs(hloc)))

def hloc_grad_psi_delta(apply_fun, H, params, sigma, h_mean, jac_mean,):
    hloc = compute_eloc(apply_fun, params, H, sigma)
    psi = apply_fun(params, sigma)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    grad = jnp.abs(jnp.sum(jac - jac_mean, axis=1))
    return jnp.log(jnp.sqrt(jnp.abs(grad))) + psi + jnp.log(jnp.sqrt(jnp.abs(hloc - h_mean)))
    

def IS_estimator_hloc(log_prob_fun, apply_fun, H, params, sigma):
    # obs: function of sigma
    log_psi = apply_fun(params, sigma)
    log_prob = log_prob_fun(params, sigma)
    w = jnp.abs(jnp.exp(log_psi))**2 / jnp.abs(jnp.exp(log_prob))**2
    Hloc = compute_eloc(apply_fun, params, H, sigma)
    norm_const = jnp.mean(w)
    
    return w*Hloc, norm_const

def IS_estimator_force_S(log_prob_fun, apply_fun, H, params, sigma):
    log_psi = apply_fun(params, sigma)
    log_prob = log_prob_fun(params, sigma)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    w = (jnp.abs(jnp.exp(log_psi))**2 / jnp.abs(jnp.exp(log_prob))**2)
    norm_const = jnp.mean(w)
    jac_c = jac - jnp.mean(w[:,None]*jac, axis=0)/norm_const
    jac_c_sqrt = jnp.sqrt(w[:,None]/len(sigma))*jac_c
    hloc = compute_eloc(apply_fun, params, H, sigma)
    hloc_c = jnp.sqrt(w/len(sigma))*(hloc - jnp.mean(w*hloc)/norm_const)
    return  (jac_c_sqrt.conj().T @ hloc_c)/norm_const, (jac_c_sqrt.conj().T @ jac_c_sqrt)/norm_const

def estimator_std_force(log_prob_fun, apply_fun, H, params, sigma):
    log_psi = apply_fun(params, sigma)
    log_prob = log_prob_fun(params, sigma)
    jac = nkjax.jacobian(
            apply_fun,
            params["params"],
            sigma, #in MC state, this is vstate.samples
            {},
            mode="holomorphic",
            dense=True,
            center=False,
            chunk_size=100,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )
    w = (jnp.abs(jnp.exp(log_psi))**2 / jnp.abs(jnp.exp(log_prob))**2)
    norm_const = jnp.mean(w)
    jac_c = jac - jnp.mean(w[:,None]*jac, axis=0)/norm_const
    jac_c_sqrt = jnp.sqrt(w[:,None]/len(sigma))*jac_c
    hloc = compute_eloc(apply_fun, params, H, sigma)
    hloc_c = jnp.sqrt(w/len(sigma))*(hloc - jnp.mean(w*hloc)/norm_const)
    force_unrolled = jac_c_sqrt.conj().T * hloc_c

    std_force = jnp.std(force_unrolled,axis=1)
    return jnp.mean(std_force)

def estimate_mean_std_IS(val, norm_const):
    val_mean = jnp.mean(val.real)
    norm_mean = jnp.mean(norm_const)
    val_estim = val_mean/norm_mean
    std_val = jnp.std(val.real)
    std_norm = jnp.std(norm_const)
    return val_estim, 1/jnp.sqrt(len(val))*(jnp.abs(val_estim)*jnp.sqrt((std_val/val_mean)**2 + (std_norm/norm_mean)**2))
