"""
Define containers used in the `RunData` object defined in `runner_np.py`
"""
ENERGY_DATA = {
    'potential_init': [],
    'potential_out': [],
    'potential_proposed': [],
    'kinetic_init': [],
    'kinetic_out': [],
    'kinetic_proposed': [],
    'hamiltonian_init': [],
    'hamiltonian_out': [],
    'hamiltonian_proposed': [],
    'exp_energy_diff': [],
}

OBSERVABLES = {
    'plaq_loss': [],
    'charge_loss': [],
    'charges': [],
    'plaqs_diffs': [],
    'dplaqs': [],
    'dcharges': [],
}


RUN_DATA = {
    'plaqs': [],
    'plaqs_diffs': [],
    'charges': [],
    'dx_proposed': [],
    'dx_out': [],
    'sumlogdet_out': [],
    'sumlogdet_proposed': [],
    'accept_prob': [],
    'forward': [],
    'xdiff_r': [],
    'vdiff_r': [],
}

REVERSE_DATA = {
    'xdiff_r': [],
    'vdiff_r': [],
}

VOLUME_DIFFS = {
    'dx_in': [],
    'dv_in': [],
    'dx_out': [],
    'dv_out': [],
}

SAMPLES = {
    'x_init': [],
    'x_out': [],
    'x_proposed': [],
    'v_init': [],
    'v_out': [],
    'v_proposed': [],
}

DICTS = {
    'run_data': RUN_DATA,
    'energy_data': ENERGY_DATA,
    'reverse_data': REVERSE_DATA,
    'volume_diffs': VOLUME_DIFFS,
    'samples_dict': SAMPLES,
}
