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
