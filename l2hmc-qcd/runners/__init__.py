"""
Define containers used in the `RunData` object defined in `runner_np.py`
"""
from config import NetWeights

__date___ = '03/19/2020'
__author__ = 'Sam Foreman'
__email__ = 'saforem2@gmail.com'
# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=inconsistent-return-statements
# pylint: disable=no-else-return
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes



NET_WEIGHTS_HMC = NetWeights(0, 0, 0, 0, 0, 0)
NET_WEIGHTS_L2HMC = NetWeights(1, 1, 1, 1, 1, 1)

#  𝐀﹙𝜉'∣𝜉﹚"
#  "𝞭x_out", "𝞭x_prop",
#  𝛅 𝛍 𝛎 𝛟 𝐭 𝐀 𝛏 𝝽 𝑨
#  𝐥 𝐨 𝐠 ⎮ ㏒ 𝐉 𝐐
#  𝐯 𝐩 𝐇 𝐫 ℐ 𝒥 𝓙
#  log⎮𝐉⎮, l𝐥𝗼𝗴⎮𝐉⎮,
#  𝐀 𝜙
#  names = ["STEP", "𝛅𝐭", "𝐀(𝛏'|𝛏)",
#           "𝛅𝛟_𝛍𝛎", "exp(𝛅𝐇)", "log⎮𝐉⎮",
#           "𝛅𝐱𝐫", "𝛅𝐯𝐫", "𝛅𝐐", "𝛅𝛟_𝐩"]
NAMES = ["step",
         "dt",
         "px",
         "dx_r",
         "dv_r",
         "sumlogdet",
         "exp(dH)",
         "ploss",
         "qloss",
         "dplaqs",
         "dQ",
         "p_err"]
         #  "𝞭𝛟_µυ",
         #  "𝞭Q",
         #  "𝞭𝛟_p",
         #  "TYPE"]


#  H0 = ["{:^13s}".format("STEP")]
HEADER = ''.join(["{:^12s}".format(name) for name in NAMES])
#  HEADER = H0 + H1

SEPERATOR = len(HEADER) * '-'
HSTR = SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR

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
