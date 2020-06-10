from runners.hmc_runner import run_hmc_loop


def main(kwargs):
    run_hmc_loop(kwargs)


if __name__ == '__main__':
    KWARGS = {
        'print_steps': 100,
        'batch_size': 32,
        'run_steps': 10000,
        'lf_arr': [2, 3, 4, 5],
        'eps_arr': [0.05, 0.075, 0.1, 0.125, 0.15],
        'beta_arr': [4.25, 4.75],
        'time_size': 16,
        'space_size': 16,
    }

    main(KWARGS)
