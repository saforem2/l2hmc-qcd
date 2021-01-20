import os

from config import PROJECT_DIR, NetWeights

import utils.file_io as io
from dynamics.gauge_dynamics import build_dynamics
from utils.inference_utils import run


def load_configs_from_log_dir(log_dir):
    configs = io.loadz(os.path.join(log_dir, 'configs.z'))
    return configs


def check_location():
    if os.path.isdir(os.path.abspath('/Users/saforem2')):
        return 'local'
    if os.path.isdir(os.path.abspath('/home/foremans')):
        return 'remote'


def run_from_log_dir(log_dir: str, net_weights: NetWeights, run_steps=5000):
    configs = load_configs_from_log_dir(log_dir)
    if 'x_shape' not in configs['dynamics_config'].keys():
        x_shape = configs['dynamics_config']['lattice_shape']
        configs['dynamics_config']['x_shape'] = x_shape

    beta = configs['beta_final']
    run_dir = os.path.join(PROJECT_DIR, 'l2hmc_function_tests',
                           'inference', f'beta{beta}')
    io.check_else_make_dir(outdir)
    log_dir = configs.log_dir
    configs['log_dir_orig'] = configs.log_dir
    configs['log_dir'] = run_dir
    configs.run_steps = run_steps

    dynamics = build_dynamics(configs)
    xnet, vnet = dynamics._load_networks(log_dir)
    dynamics.xnet = xnet
    dynamics.vnet = vnet
    dynamics.net_weights = net_weights
    inference_results = run(dynamics, configs, beta=beta, runs_dir=run_dir,
                            make_plots=True, therm_frac=0.2, num_chains=16)

    return inference_results


def main():
    location = check_location()
    if location == 'local':
        base_dir = os.path.abspath('/Users/saforem2/thetaGPU/training')
    else:
        base_dir = os.path.abspath('/home/foremans/thetaGPU/training')

    log_dirs = [
        os.path.join(base_dir,
                     '2020_11_30/t16x16_b2048_lf10_bi5_bf6_4ranks/l2hmc-qcd/logs/'
                     'GaugeModel_logs/2020_11/L16_b2048_lf10_bi5_bf6_dp02_clip500_sepNets_NCProj/'),
        os.path.join(base_dir,
                     '2020_12_02/t16x16_b2048_lf10_bi6_bf7_4ranks/l2hmc-qcd/logs/'
                     'GaugeModel_logs/2020_12/L16_b2048_lf10_bi6_bf7_dp02_clip500_sepNets_NCProj/'),
        os.path.join(base_dir,
                     '2021_01_02/t16x16b2048lf20bi4bf5/l2hmc-qcd/logs/'
                     'GaugeModel_logs/2021_01/L16_b2048_lf20_bi4_bf5_dp02_clip500_sepNets_NCProj/'),
        os.path.join(base_dir,
                     '2021_01_06/t16x16b2048lf10bi3bf5/l2hmc-qcd/logs/'
                     'GaugeModel_logs/2021_01/L16_b2048_lf10_bi3_bf5_dp02_clip500_sepNets_NCProj/'),
    ]

    nw_arr = [
        NetWeights(0., 1., 1., 1., 1., 1.),
        NetWeights(1., 0., 1., 1., 1., 1.),
        NetWeights(1., 1., 0., 1., 1., 1.),
        NetWeights(1., 1., 1., 0., 1., 1.),
        NetWeights(1., 1., 1., 1., 0., 1.),
        NetWeights(1., 1., 1., 1., 1., 0.),
    ]

    for log_dir in log_dirs:
        for nw in nw_arr:
            _ = run_from_log_dir(log_dir, nw)

    return


if __name__ == '__main__':
    main()
