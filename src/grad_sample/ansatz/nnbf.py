import flax.linen as nn
import netket as nk
import jax.numpy as jnp
from netket.nn.masked_linear import default_kernel_init
from typing import Any
import jax

DType = Any

class LogNeuralBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    hidden_units: int
    kernel_init: Any = default_kernel_init
    param_dtype: Any = jnp.float32

    def setup(self):
        """Initialize model parameters."""
        # The N x Nf matrix of the orbitals
        self.M = self.param(
            "M", self.kernel_init, 
            (2 * self.hilbert.n_orbitals, self.hilbert.n_fermions), 
            self.param_dtype
        )

        # Construct the Backflow: Takes (N,) occupation numbers -> (N, Nf) orbital transformation matrix
        self.backflow = nn.Sequential([
            nn.Dense(features=self.hidden_units, param_dtype=self.param_dtype),
            nn.tanh,
            nn.Dense(features=2 * self.hilbert.n_orbitals * self.hilbert.n_fermions, param_dtype=self.param_dtype),
            lambda x: x.reshape(x.shape[:-1] + (2 * self.hilbert.n_orbitals, self.hilbert.n_fermions))
        ])

    def log_sd(self, n: jax.Array) -> jax.Array:
        """Compute the log of the Slater determinant with backflow for a single input sample."""
        # Compute backflow correction
        F = self.backflow(n)
        M = self.M + F

        # Find occupied orbitals
        R = n.nonzero(size=self.hilbert.n_fermions)[0]
        A = M[R]
        return nk.jax.logdet_cmplx(A)

    def __call__(self, n: jax.Array) -> jax.Array:
        """Vectorized computation over batches."""
        return jax.vmap(self.log_sd)(n)