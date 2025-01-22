import jax
import jax.numpy as jnp

def shape_tree(pytree):
    return jax.tree_util.tree_map(lambda g: g.shape, pytree)

def snr_tree(pytree, pytree_fs):
    #centered on the full summation value
    return jax.tree_util.tree_map(lambda g, g_fs: jnp.sqrt(g.shape[0]*jnp.abs(g_fs)**2/(jnp.var(g, axis=0))), pytree, pytree_fs)

def flatten_tree_to_array(tree):
    """
    Flattens a pytree where each leaf is an array with the first dimension `N`.
    Produces a single array of shape (N, N_elem), where N is the batch dimension
    and N_elem is the sum of flattened dimensions of all leaves (excluding N).

    Parameters:
    - tree: The input pytree (e.g., nested dicts, lists, or tuples of arrays).

    Returns:
    - A jax.numpy.ndarray of shape (N, N_elem).
    """
    # Flatten the tree and extract leaves
    leaves, _ = jax.tree_util.tree_flatten(tree)
    
    # Ensure all leaves share the same batch dimension N
    batch_sizes = [leaf.shape[0] for leaf in leaves]
    if len(set(batch_sizes)) > 1:
        raise ValueError("All leaves must have the same first batch dimension (N).")
    
    N = batch_sizes[0]  # Shared batch dimension
    
    # Flatten each leaf (excluding the batch dimension) and concatenate
    flattened_leaves = [leaf.reshape(N, -1) for leaf in leaves]
    return jnp.concatenate(flattened_leaves, axis=1)