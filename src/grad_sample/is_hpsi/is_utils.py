
from netket_pro.utils import make_logpsi_U_afun, make_logpsi_sum_afun

def _prepare_H(log_psi, log_psi_variables, op):
    # use nkpro utils to compute hpsi wavefunction, sum disabled for now
    log_Hpsi, log_Hpsi_variables = make_logpsi_U_afun(log_psi, op.operator, log_psi_variables)

    # logpsi_diff_fun, new_variables = make_logpsi_sum_afun(
    #     log_psi,
    #     log_Hpsi,
    #     log_ψ_variables,
    #     log_Hpsi_variables,
    #     epsilon=op.epsilon,
    # )

    return log_Hpsi, log_Hpsi_variables