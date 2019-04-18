"""
runner.py

Implements GaugeModelRunner class responsible for running the L2HMC algorithm
on a U(1) gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import time
import pickle
from scipy.stats import sem
from collections import Counter, OrderedDict
import numpy as np

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from globals import NP_FLOAT


class GaugeModelRunner:
    def __init__(self, sess, model, logger):
        self.sess = sess
        self.model = model
        self.logger = logger
        #  self.plotter = plotter

    def _save(self, run_data, run_strings, **kwargs):
        """Update logger and save run information."""
        therm_frac = kwargs['therm_frac']
        run_dir = kwargs['run_dir']
        stats_file = kwargs['stats_file']
        run_file = kwargs['run_file']
        data_file = kwargs['data_file']
        stats_data_file = kwargs['stats_data_file']

        io.log(f"Saving run_data to: {data_file}.")
        with open(data_file, 'wb') as f:
            pickle.dump(run_data, f)

        stats = self.calc_observables_stats(run_data, therm_frac)

        io.log(f"Saving run_data_stats to: {stats_data_file}.")
        with open(stats_data_file, 'wb') as f:
            pickle.dump(stats, f)

        self.logger.save_run_info(run_data, stats, run_dir, **kwargs)
        self.logger.write_run_stats(run_data, stats, stats_file, **kwargs)
        self.logger.write_run_strings(run_strings, run_file)

    def run_step(self, step, run_steps, inputs):
        """Perform a single run step.

        Args:
            step (int): Current step.
            run_steps (int): Total number of run_steps to perform.
            inputs (tuple): Tuple consisting of 
                (samples_np, beta_np, eps, plaq_exact)
                where samples_np is the input batch of samples, beta_np is the
                input value of beta, eps is the step size, and plaq_exact is
                the expected (theoretical) value of the avg. plaquette at that
                particular beta.
        Returns:
            outputs (tuple): Tuple of outputs consisting of 
                (new_samples, accept_prob, actions, charges, charge_diffs).
                Where new_samples has the same shape as input_samples;
                accept_prob, actions, charges, and charge diffs all have shape
                (model.batch_size,)
        """
        samples_np, beta_np, eps, plaq_exact = inputs

        feed_dict = {
            self.model.x: samples_np,
            self.model.beta: beta_np
        }

        ops = [
            self.model.x_out,
            self.model.px,
            self.model.actions_op,
            self.model.plaqs_op,
            self.model.charges_op,
            self.model.charge_diffs_op
        ]

        start_time = time.time()

        outputs = self.sess.run(ops, feed_dict=feed_dict)

        dt = time.time() - start_time

        out_data = {
            'samples': np.mod(outputs[0], 2 * np.pi),
            'px': outputs[1],
            'actions': outputs[2],
            'plaqs': outputs[3],
            'charges': outputs[4],
            'charge_diffs': outputs[5]
        }

        data_str = (f'{step:>5g}/{run_steps:<6g} '
                    f'{dt:^9.4g} '                      # time / step
                    f'{np.mean(outputs[1]):^9.4g} '     # accept. prob
                    f'{eps:^9.4g} '                     # step size
                    f'{beta_np:^9.4g} '                 # beta val
                    f'{np.mean(outputs[2]):^9.4g} '     # avg. actions
                    f'{np.mean(outputs[3]):^9.4g} '     # avg. plaquettes
                    f'{plaq_exact:^9.4g} '              # exact plaquette val
                    f'{outputs[5]:^9.4g} ')             # top. charge diff

        return out_data, data_str

    def run(self,
            run_steps,
            current_step=None,
            beta_np=None,
            therm_frac=10,
            ret=False):
        """Run the simulation to generate samples and calculate observables.

        Args:
            run_steps: Number of steps to run the sampler for.
            current_step: Integer passed when the sampler is ran intermittently
                during training, as a way to monitor the models performance
                during training. By passing the current training step as
                current_step, this data is saved to a unique directory labeled
                by the current_step.
            beta: Float value indicating the inverse coupling constant that the
                sampler is to be run at.

        Returns:
            observables: Tuple of observables dictionaries consisting of:
                (actions_dict, plaqs_dict, charges_dict, charge_diffs_dict).
        """

        if not isinstance(run_steps, int):
            run_steps = int(run_steps)

        if beta_np is None:
            beta_np = self.model.beta_final

        #  run_key = (run_steps, beta_np)
        charges_arr = []
        run_strings = []
        run_data = {
            'px': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diffs': {},
        }

        run_params = {
            'run_steps': run_steps,
            'beta': beta_np,
        }

        if current_step is None:
            _dir = self.logger.runs_dir
            txt_file = f'run_info_steps_{run_steps}_beta_{beta_np}.txt'
            stats_file = f'run_stats_steps_{run_steps}_beta_{beta_np}.txt'
            training = False
        else:
            _dir = self.logger.train_runs_dir
            txt_file = (f"run_info_{current_step}_TRAIN_"
                        f"steps_{run_steps}_beta_{beta_np}.txt")
            stats_file = (f"run_stats_{current_step}_TRAIN_"
                          f"steps_{run_steps}_beta_{beta_np}.txt")
            training = True
            # -----------------------------------------------
            # set to False so batch norm params don't change
            # -----------------------------------------------
            self.model.dynamics.trainable = False

        _dir_name = f"steps_{run_steps}_beta_{beta_np}"
        run_dir = os.path.join(_dir, _dir_name)

        data_file = os.path.join(run_dir, 'run_data.pkl')
        stats_data_file = os.path.join(run_dir, 'run_data_stats.pkl')
        run_file = os.path.join(run_dir, txt_file)
        stats_file = os.path.join(run_dir, stats_file)

        eps = self.sess.run(self.model.dynamics.eps)
        plaq_exact = u1_plaq_exact(beta_np)

        # start with randomly generated samples
        samples_np = np.random.randn(*(self.model.batch_size,
                                       self.model.x_dim))

        try:
            for step in range(run_steps):
                inputs = (samples_np, beta_np, eps, plaq_exact)
                out_data, data_str = self.run_step(step, run_steps, inputs)
                self.logger.update_running(step, out_data,
                                           data_str, run_params)

                key = (step, beta_np)

                samples_np = out_data['samples']

                run_data['px'][key] = out_data['px']
                run_data['actions'][key] = out_data['actions']
                run_data['plaqs'][key] = out_data['plaqs']
                run_data['charges'][key] = out_data['charges']
                run_data['charge_diffs'][key] = out_data['charge_diffs']
                charges_arr.append(out_data['charges'])

                run_strings.append(data_str)

            kwargs = {
                'run_dir': run_dir,
                'stats_file': stats_file,
                'run_file': run_file,
                'data_file': data_file,
                'stats_data_file': stats_data_file,
                'run_steps': run_steps,
                'current_step': current_step,
                'beta': beta_np,
                'therm_frac': therm_frac,
                'training': training,
                'therm_steps': run_steps // therm_frac,
            }
            self._save(run_data, run_strings, **kwargs)

            if training:
                self.model.dynamics.trainable = True

            if ret:
                return run_data

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected!")
            io.log("Saving current state and exiting.")
            self._save(run_data, run_strings, **kwargs)
            if training:
                self.model.dynamics.trainable = True
            if ret:
                return run_data

    def calc_observables_stats(self, run_data, therm_frac=10):
        """Calculate statistics for lattice observables.

        Args:
            run_data: Dictionary of observables data. Keys denote the
                observables name.
            therm_frac: Fraction of data to throw out for thermalization.

        Returns:
            stats: Dictionary containing statistics for each observable in
            run_data. Additionally, contains `charge_probs` which is a
            dictionary of the form {charge_val: charge_val_probability}.
        """
        def get_stats(data, t_frac=10):
            if isinstance(data, dict):
                arr = np.array(list(data.values()))
            elif isinstance(data, (list, np.ndarray)):
                arr = np.array(data)

            num_steps = arr.shape[0]
            therm_steps = num_steps // t_frac
            arr = arr[therm_steps:, :]
            avg = np.mean(arr, axis=0)
            err = sem(arr)

            return (avg, err)

        actions_stats = get_stats(run_data['actions'], therm_frac)
        plaqs_stats = get_stats(run_data['plaqs'], therm_frac)

        charges_arr = np.array(list(run_data['charges'].values()), dtype=int)
        charges_stats = get_stats(charges_arr, therm_frac)

        suscept_arr = charges_arr ** 2
        suscept_stats = get_stats(suscept_arr)

        charge_probs = {}
        counts = Counter(list(charges_arr.flatten()))
        total_counts = np.sum(list(counts.values()))
        for key, val in counts.items():
            charge_probs[key] = val / total_counts

        charge_probs = OrderedDict(sorted(charge_probs.items(),
                                          key=lambda k: k[0]))

        stats = {
            'actions': actions_stats,
            'plaqs': plaqs_stats,
            'charges': charges_stats,
            'suscept': suscept_stats,
            'charge_probs': charge_probs
        }

        return stats
