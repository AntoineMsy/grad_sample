import jax
import jax.numpy as jnp

def shape_tree(pytree):
    return jax.tree_util.tree_map(lambda g: g.shape, pytree)

def snr_tree(pytree):
    return jax.tree_util.tree_map(lambda g: jnp.sqrt(jnp.abs(jnp.mean(g))/(g.size*jnp.var(g))), pytree)