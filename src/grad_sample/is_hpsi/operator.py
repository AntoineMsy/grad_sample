from typing import Optional

from netket.experimental.observable import AbstractObservable
from netket.operator import DiscreteJaxOperator

from jax.tree_util import register_pytree_node_class

from grad_sample.is_hpsi.is_utils import _prepare_H, make_logpsi_smeared_afun
import jax.numpy as jnp

@register_pytree_node_class
class IS_Operator(AbstractObservable):
    """ """
    def __init__(
        self,
        operator: DiscreteJaxOperator,
        is_mode: float,
        mode = 'holomorphic',
        *,
        resample_fraction: Optional[int] = None,
    ):
        """

        Args:
            operator: The operator to estimate with importance sampling
            is_mode: the probability distribution to use for IS, -1.0 being Hpsi, beetween 0 and 2 being smearing
            mode: how to compute the jacobians ; either holomorphic or real
            resample_fraction: Resample only a fraction of samples. None or
                1 to disable. 0 to never resample after the first iteration.
        """
        if not isinstance(operator, DiscreteJaxOperator):
            if hasattr(operator, "to_jax_operator"):
                operator = operator.to_jax_operator()
            else:
                raise TypeError("Only jax operators supported.")

        super().__init__(operator.hilbert)
        self._bare_operator = operator

        self._resample_fraction = resample_fraction
        self._is_mode = is_mode

        if mode not in ['holomorphic', 'real']:
            raise ValueError('Invalid jacobian mode specified. Must be either real or holomorphic')
        
        self._mode = mode

    @property
    def operator(self):
        return self._bare_operator

    @property
    def resample_fraction(self):
        return self._resample_fraction

    @property
    def is_hermitian(self):
        return self.operator.is_hermitian

    @property
    def is_mode(self):
        return self._is_mode
    
    @property
    def mode(self):
        return self._mode

    def collect(self):
        return self

    def tree_flatten(self):
        children = (
            self.operator,
            self.resample_fraction
        )
        aux_data = {'is_mode' : self.is_mode, 'mode': self.mode}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # square_fast = aux_data.pop("square_fast")
        is_mode = aux_data.pop('is_mode')
        mode = aux_data.pop('mode')
        (operator, resample_fraction) = children
        
        res = cls(
            operator,
            resample_fraction=resample_fraction,
            is_mode = is_mode,
            mode=mode
        )
        res._is_mode = is_mode
        res._mode = mode
        return res

    def get_log_importance(self, vstate):
        if self.is_mode == -1: #-1 stands for hpsi
            return _prepare_H(vstate._apply_fun, vstate.variables, self)
        elif isinstance(self.is_mode, float) or jnp.issubdtype(self.is_mode.dtype, jnp.floating):
            return make_logpsi_smeared_afun(vstate._apply_fun, vstate.variables, self.is_mode)
        else:
            print("invalide IS mode specified")
