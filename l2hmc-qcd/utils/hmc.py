"""
hmc.py

Implements methods for running and training generic HMC.
"""
from __future__ import absolute_import, print_function, division, annotations

def _get_hmc_log_str(configs):
    dynamics_config = configs.get('dynamics_config', None)

    lf = dynamics_config.get('num_steps', None)
    eps = dynamics_config.get('eps', None)
    ls = dynamics_config.get('x_shape', None)
    bs = ls[0]  # batch size
    nx = ls[1]  # size in 'x' direction

    b = configs.get('beta', None)
    if b is None:
        b = configs.get('beta_final', None)

    log_str = (
        f'HMC_L{nx}_b{bs}_beta{float(b)}_lf{lf}_eps{eps}'.replace('.0', '')
    )

    log_str = log_str.replace('.', '')

    return log_str


def run_hmc(
        configs: dict[str, Any],
        hmc_dir: str = None,
        skip_existing: bool = False,
        save_x: bool = False,
        therm_frac: float = 0.33,
        num_chains: int = 16,
        make_plots: bool = True,
) -> InferenceResults:
    """Run HMC using `inference_args` on a model specified by `params`.

    NOTE:
    -----
    args should be a dict with the following keys:
        - 'hmc'
        - 'eps'
        - 'beta'
        - 'num_steps'
        - 'run_steps'
        - 'x_shape'
    """
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None, None)

    if hmc_dir is None:
        month_str = io.get_timestamp('%Y_%m')
        hmc_dir = os.path.join(HMC_LOGS_DIR, month_str)

    io.check_else_make_dir(hmc_dir)

    if skip_existing:
        fstr = io.get_run_dir_fstr(configs)
        base_dir = os.path.dirname(hmc_dir)
        matches = list(
            Path(base_dir).rglob(f'*{fstr}*')
        )
        if len(matches) > 0:
            logger.warning('Existing run with current parameters found!')
            logger.print_dict(configs)
            return InferenceResults(None, None, None, None, None)

    dynamics = build_dynamics(configs)
    try:
        inference_results = run(dynamics=dynamics, configs=configs,
                                runs_dir=hmc_dir, make_plots=make_plots,
                                save_x=save_x, therm_frac=therm_frac,
                                num_chains=num_chains)
    except FileExistsError:
        inference_results = InferenceResults(None, None, None, None, None)
        logger.warning('Existing run with current parameters found! Skipping!')

    return inference_results


