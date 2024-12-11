from typing import Optional

from netket.experimental.observable import AbstractObservable
from netket.operator import DiscreteJaxOperator

from jax.tree_util import register_pytree_node_class

from grad_sample.is_hpsi.is_utils import _prepare_H


@register_pytree_node_class
class IS_Operator(AbstractObservable):
    """ """

    def __init__(
        self,
        operator: DiscreteJaxOperator,
        *,
        # second_order: bool = True,
        # square_fast: bool = False,
        resample_fraction: Optional[int] = None,
        # reweight_norm: bool = True,
    ):
        """

        Args:
            operator: The operator to estimate with importance sampling
            epsilon: The :math:`shift` used to sample
            second_order: Whether to keep second order terms in epsilon.
                False by default
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
        # self._epsilon = epsilon
        # self._second_order = second_order
        self._resample_fraction = resample_fraction
        # self._reweight_norm = reweight_norm

        # self._square_fast = square_fast
        # if square_fast:
        #     self._operator_squared = None
        # else:
        op_sq = operator.H @ operator
        if not isinstance(op_sq, DiscreteJaxOperator):
            op_sq = op_sq.to_jax_operator()
        self._operator_squared = op_sq

    """
    @property
    def reweight_norm(self):
        return self._reweight_norm

    @property
    def second_order(self):
        return self._second_order
    """

    # @property
    # def epsilon(self):
    #     return self._epsilon

    @property
    def operator(self):
        return self._bare_operator

    @property
    def resample_fraction(self):
        return self._resample_fraction

    @property
    def is_hermitian(self):
        return self.operator.is_hermitian

    # @property
    # def square_fast(self):
    #     return self._square_fast

    def collect(self):
        return self

    def tree_flatten(self):
        children = (
            self.operator,
            # self.epsilon,
            self.resample_fraction,
            self._operator_squared,
        )
        # aux_data = {"second_order": self.second_order, "square_fast": self.square_fast, "reweight_norm": self.reweight_norm}
        # aux_data = {"square_fast": self.square_fast}
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # square_fast = aux_data.pop("square_fast")

        # (operator, epsilon, resample_fraction, op_sq) = children
        (operator, op_sq) = children
        res = cls(
            operator,
            # epsilon=epsilon,
            resample_fraction=resample_fraction,
            **aux_data,
            # square_fast=True,
        )
        # res._square_fast = square_fast
        res._operator_squared = op_sq
        return res

    def get_log_importance(self, vstate):
        return _prepare_H(vstate._apply_fun, vstate.variables, self)
