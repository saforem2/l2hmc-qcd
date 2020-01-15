import sys
import argparse
import shlex


def parse_args():
    """Parse command line arguments."""
    description = 'Run tf-independent inference on trained model.'
    parser = argparse.ArgumentParser(
        description=description,
        fromfile_prefix_chars='@',
    )

    parser.add_argument('--log_dir',
                        dest='log_dir',
                        required=True,
                        default=None,
                        help=("""log_dir containing `weights.pkl` file of
                              trained models' network weights."""))

    parser.add_argument('--hmc',
                        dest='hmc',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed sets all `net_weights`
                              to 0."""))

    parser.add_argument('--num_steps',
                        dest='num_steps',
                        required=False,
                        default=None,
                        help=("""Number of leapfrog steps
                              (i.e. trajectory length)."""))

    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=None,
                        required=False,
                        help=("""Number of samples (batch size) to use for
                              training.\n (Default: 20)"""))

    parser.add_argument('--eps',
                        dest='eps',
                        type=float,
                        default=None,
                        required=False,
                        help=("""Step size to use during inference. If no value
                              is passed, `eps = None` and the optimal step size
                              (determined during training) will be used."""))

    parser.add_argument('--init',
                        dest='init',
                        type=str,
                        default=None,
                        required=False,
                        help=("""String specifying how to initialize samples
                              when running inference. Possible values are:
                              'zeros', 'ones', or 'random'.
                              (Default: 'random')"""))

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=10000,
                        required=False,
                        help=("""Number of evaluation 'run' steps to perform
                              after training (i.e. length of desired chain
                              generate using trained L2HMC sample).
                              (Default: 5000)"""))

    parser.add_argument("--beta",
                        dest="beta",
                        type=float,
                        default=5.,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run inference using the trained
                              L2HMC sampler. (Default: None"""))

    parser.add_argument('--x_scale_weight',
                        dest='x_scale_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--x_translation_weight',
                        dest='x_translation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--x_transformation_weight',
                        dest='x_transformation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the
                              `transformation_weight` parameter, a
                              multiplicative weight that scales the
                              contribution of the `transformation` (Q) function
                              when performing the augmented L2HMC molecular
                              dynamics update."""))

    parser.add_argument('--v_scale_weight',
                        dest='v_scale_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--v_translation_weight',
                        dest='v_translation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--v_transformation_weight',
                        dest='v_transformation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the
                              `transformation_weight` parameter, a
                              multiplicative weight that scales the
                              contribution of the `transformation` (Q) function
                              when performing the augmented L2HMC molecular
                              dynamics update."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
