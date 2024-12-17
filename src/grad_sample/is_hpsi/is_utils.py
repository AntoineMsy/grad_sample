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


def make_logpsi_smeared_afun(
    logpsi_fun, variables, alpha
):
    # wrap apply_fun into logpsi logpsi_U
    logpsi_smeared_fun = nkjax.HashablePartial(
        _logpsi_smeared_fun, logpsi_fun )

    new_variables = flax.core.copy(
        variables,
        {"alpha": alpha},
    )

    return logpsi_smeared_fun, new_variables

def _logpsi_smeared_fun(afun, variables, x):
    variables_afun, epsilon = flax.core.pop(variables, "alpha")

    if alpha is None:
        alpha = 1

    logpsi1_x = afun(variables_afun, x)

    return jnp.pow(jnp.abs(logpsi1_x), alpha/2)