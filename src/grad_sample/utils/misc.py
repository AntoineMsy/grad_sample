import netket as nk
from netket.operator.spin import sigmax, sigmaz
import jax.numpy as jnp
import jax
from functools import partial

# Example usage:
# directory_path = './logs'  # Change this to your actual directory path
# base_name = 'name'

# new_run_name = get_unique_run_name_from_logs(directory_path, base_name)
# print(f"Unique run name: {new_run_name}")

def get_tfi_ham(hi, graph, N, V=1):
    Gamma = 3.044*V
    H = sum([V * sigmaz(hi, i) * sigmaz(hi, j) for (i, j) in graph.edges()])
    Gamma = 3.044 * V
    H += sum([-Gamma * sigmax(hi, i) for i in range(N)])
    H_jax = H.to_pauli_strings().to_jax_operator()
    return H, H_jax


def to_array(model, parameters, all_configs):
    # begin by generating all configurations in the hilbert space.
    # all_States returns a batch of configurations that is (hi.n_states, N) large.
    # now evaluate the model, and convert to a normalised wavefunction.
    logpsi = model.apply(parameters, all_configs)
    psi = jnp.exp(logpsi)
    psi = psi / jnp.linalg.norm(psi)
    return psi

def compute_energy(model, parameters, all_configs, hamiltonian_sparse):
    psi_gs = to_array(model, parameters, all_configs)
    return psi_gs.conj().T@(hamiltonian_sparse@psi_gs)

compute_energy_jit = jax.jit(compute_energy, static_argnames="model")

@partial(jax.jit, static_argnames='model')
def compute_jacobian(model, parameters, hamiltonian_sparse, s):
       # compile the function and make it faster to execute
    # reshape the samples, the samples are divided in different Markov chains
    s = s.reshape(-1, s.shape[-1])

    # compute the dk logpsi
    logpsi_fun = lambda pars : model.apply(pars, s)    # we freeze the samples so the functions depend only on the parameters
    jacobian =  jax.jacrev(logpsi_fun, holomorphic=True)(parameters)  # compute the jacobian of the log of the (holomorphic) wavefunction

    return jacobian

from functools import partial 
@partial(jax.jit, static_argnames='model')   # compile the function and make it faster to execute
def compute_eloc(model, parameters, ham, s):
    # reshape the samples to have shape (n_samples, N), the samples are divided in different Markov chains
    s = s.reshape(-1, s.shape[-1])

    # compute the connected configurations and the matrix elements
    eta, eta_H_sigma = ham.get_conn_padded(s)

    # compute the local energies (in log-spacde for numerical stability)
    logpsi_eta = model.apply(parameters,eta)    # evaluate the wf on the samples
    logpsi_sigma = model.apply(parameters, s)  # evaluate the wf on the connected configurations
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1) # add a dimension to match the shape of logpsi_eta
    E_loc =  jnp.sum(eta_H_sigma * jnp.exp(logpsi_eta -logpsi_sigma), axis=-1)     # compute the local energies

    return E_loc


@partial(jax.jit, static_argnames='model')   # compile the function and make it faster to execute
def estimate_energy_and_gradient(model, parameters, ham, s):
    # reshape the samples, the samples are divided in different Markov chains
    s = s.reshape(-1, s.shape[-1])

    # compute eloc
    E_loc = compute_eloc(model, parameters, ham, s)

    # compute jacobian
    jacobian = compute_jacobian(model, parameters, ham, s)

    # take the number of samples
    n_samples = E_loc.shape[0]

    # compute the energy
    E_average = jnp.mean(E_loc)
    E_variance = jnp.std(E_loc)
    E_error = jnp.sqrt(E_variance/n_samples)
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)  # create a Netket object containing the statistics

    # center the local energies
    E_loc -= E_average
    # compute the gradient as Ok.conj() @ E_loc / n_samples (operate on pytree with jax.tree.map) 
    E_grad = jax.tree.map(lambda jac: jnp.einsum("i...,i",jac.conj(), E_loc)/ n_samples, jacobian)   
    
    # comptue the gradient ...
    # first define the function to be differentiated
    # logpsi_sigma_fun = lambda pars : model.apply(pars, s)

    # # use jacrev with jax.tree.map, or even better, jax.vjp
    # _, vjpfun = nk.jax.vjp(logpsi_sigma_fun, parameters, conjugate=True)
    # # E_grad = vjpfun((E_loc.conj())/E_loc.size)[0]
    # # print(E_grad_tree, E_grad)
    return E, E_grad

def flatten_jacobian(pytree):
    # Apply reshape operation to each leaf of the pytree
    reshaped_pytree = jax.tree.map(lambda x: jnp.reshape(x, (x.shape[0], -1)), pytree)
    
    # Convert the pytree to a flat list of arrays
    flat_list, _ = jax.tree_util.tree_flatten(reshaped_pytree)
    
    # Stack the arrays in the list along the second dimension
    result = jnp.concatenate(flat_list, axis=-1)
    
    return result

def SR(E_grad, jacobian):

    # convert from pytree to dense array
    E_grad_dense, unravel = nk.jax.tree_ravel(E_grad)
    jacobian_dense = flatten_jacobian(jacobian)

    # take the number of samples
    n_configs = jacobian_dense.shape[0]

    # center the jacobians
    jacobian_centered = jacobian_dense - jnp.mean(jacobian_dense,axis=0) #(N_s, N*N_conn*2)
    # w = jacobian_centered @ E_grad
    # res = jnp.tensordot(w.conj(),jacobian_centered, axes=w.ndim).conj()
    # compute the S matrix
    S = jacobian_centered.T.conj() @ jacobian_centered / n_configs
    # print(S.shape)
    # condition the S matrix 
    S = S + 0.00001 * jnp.eye(S.shape[0])
    # solve the linear system (use the system jax.scipy.sparse.linalg.cg)
    E_grad_nat = jax.scipy.sparse.linalg.cg(S, E_grad_dense)[0]
    
    return unravel(E_grad_nat)