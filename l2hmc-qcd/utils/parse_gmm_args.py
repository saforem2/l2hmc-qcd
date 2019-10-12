"""
parse_gmm_args.py

Implements method for parsing command line arguments for `gmm_main.py`.

Author: Sam Foreman (github: @saforem2)
Date: 09/18/2019
"""
import sys
import argparse
import shlex

DESCRIPTION = (
    'Implementation of the L2HMC algorithm for the 2D Gaussian Mixture Model.'
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        fromfile_prefix_chars='@',
    )

    parser.add_argument('--center',
                        dest='center',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Float value specifying the center of the
                              target distribution. For example, if `--center=1`
                              is passed, the target distribution will have
                              means located at [-1, 0], and  [0, 1]. (Default:
                              1.)"""))

    parser.add_argument('--arrangement',
                        dest='arrangement',
                        type=str,
                        default='xaxis',
                        required=False,
                        help=("""Arrangement of the means of the target
                              distribution. A string for specifying
                              alternative arrangements for the target
                              distribution. Possible values are: `xaxis`,
                              `yaxis`, `diag`, `lattice`, or `ring`."""))

    parser.add_argument('--size',
                        dest='size',
                        default=1.,
                        type=float,
                        required=False,
                        help=("""The `size` of the distribution. Only relevant
                              when `--arrangement == 'lattice' or 'ring'`. In
                              the `lattice` case, specifies the linear extent
                              of the lattice. In the `ring` case, specifies the
                              radius of the ring."""))

    parser.add_argument('--diag',
                        dest='diag',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed aligns the centers of
                              each of the component distributions along a
                              diagonal axis instead of along the x-axis."""))

    parser.add_argument('--num_distributions',
                        dest='num_distributions',
                        type=int,
                        default=2,
                        required=False,
                        help=("""Number of distributions to include for
                              Gaussian Mixture Model, (Default: 2)."""))

    parser.add_argument('--sigma1',
                        dest='sigma1',
                        type=float,
                        default=0.02,
                        required=False,
                        help=("""Variance of first distribution in GMM
                              model. (Default: 0.02)"""))

    parser.add_argument('--sigma2',
                        dest='sigma2',
                        type=float,
                        default=0.02,
                        required=False,
                        help=("""Variance of first distribution in GMM
                              model. (Default: 0.02)"""))

    parser.add_argument('--num_steps',
                        dest='num_steps',
                        type=int,
                        default=10,
                        required=False,
                        help=("""Number of leapfrog steps to use in the
                              augmented molecular dynamics integration for the
                              L2HMC algorithm. (Default: 10)"""))

    parser.add_argument('--eps',
                        dest='eps',
                        type=float,
                        default=0.2,
                        required=False,
                        help=("""Initial value of the step size to use in the
                              Molecular Dynamics (MD) update. (Default:
                              0.2)"""))

    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=128,
                        required=False,
                        help=("""Number of chains to run in parallel, i.e.a
                              single batch consists of `batch_size` independent
                              chains. (Default: 128)"""))

    parser.add_argument('--use_gaussian_loss',
                        dest='use_gaussian_loss',
                        action='store_true',
                        required=False,
                        help=("""If passed, set `use_gaussian_loss=True` and
                              use alternative gaussian loss function."""))

    parser.add_argument('--use_nnehmc_loss',
                        dest='use_nnehmc_loss',
                        action='store_true',
                        required=False,
                        help=("""If passed, set `use_nnehmc_loss=True` and
                              use alternative NNEHMC loss function."""))

    parser.add_argument('--nnehmc_beta',
                        dest='nnehmc_beta',
                        default=1.,
                        required=False,
                        help=("""Value of `beta` to use when calculating the
                              NNEHMC loss. Note that beta multiplies the HMC
                              acceptance probability in Eq. 14. of
                              https://link.springer.com/chapter/10.1007/978-3-030-20351-1_64
                              (Default: 1., but only applies when
                              `--use_nnehmc_loss` is True."""))  # noqa: E501

    parser.add_argument("--aux_weight",
                        dest="aux_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of auxiliary term in loss function.
                              (Default: 1.)"""))

    parser.add_argument('--loss_scale',
                        dest='loss_scale',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Loss scaling factor. A multiplicative
                              constant that scales the overall value of the
                              total loss."""))

    parser.add_argument('--beta_init',
                        dest='beta_init',
                        type=float,
                        default=1./10,
                        required=False,
                        help=("""Initial value of beta (inverse temperature) to
                              use when training begins. Since the training
                              procedure uses simulated annealing, `beta` runs
                              from `beta_init` to `beta_final`. Note that
                              `beta_final=1` for the GMM model."""))

    parser.add_argument('--beta_final',
                        dest='beta_final',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Final value of beta (inverse temperature) that
                              will occur at the end of the training phase.
                              (Default: 1)"""))

    parser.add_argument('--num_hidden1',
                        dest='num_hidden1',
                        type=int,
                        default=50,
                        required=False,
                        help=("""Number of hidden nodes to use in first group
                              of layers in neural network to be trained.
                              (Default: 50)"""))

    parser.add_argument('--num_hidden2',
                        dest='num_hidden2',
                        type=int,
                        default=50,
                        required=False,
                        help=("""Number of hidden nodes to use in second group
                              of layers in neural network to be trained.
                              (Default: 50)"""))

    parser.add_argument("--train_steps",
                        dest="train_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of training steps to perform.
                              (Default: 5000)"""))

    parser.add_argument('--lr_init',
                        dest='lr_init',
                        type=float,
                        default=0.001,
                        help=("""Initial value of the learning rate to
                              use."""))

    parser.add_argument("--lr_decay_steps",
                        dest="lr_decay_steps",
                        type=int,
                        default=500,
                        required=False,
                        help=("""Number of steps after which to decay learning
                              rate. (Default: 500)"""))

    parser.add_argument("--lr_decay_rate",
                        dest="lr_decay_rate",
                        type=float, default=0.96,
                        required=False,
                        help=("""Learning rate decay rate to be used during
                              training. (Default: 0.96)"""))

    parser.add_argument('--no_summaries',
                        dest="no_summaries",
                        action="store_true",
                        required=False,
                        help=("""FLag that when passed will prevent tensorflow
                              from creating tensorboard summary objects."""))

    parser.add_argument("--global_seed",
                        dest='global_seed',
                        type=int,
                        default=42,
                        required=False,
                        help=("""Sets global seed to ensure
                              reproducibility."""))

    parser.add_argument('--save_lf',
                        dest='save_lf',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will save the
                              output from each leapfrog step."""))

    parser.add_argument("--float64",
                        dest="float64",
                        action="store_true",
                        required=False,
                        help=("""When passed, using 64 point floating precision
                              by settings globals.TF_FLOAT = tf.float64. False
                              by default (use tf.float32)."""))

    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed indicates we're training
                              using an NVIDIA GPU."""))

    parser.add_argument("--horovod",
                        dest="horovod",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed uses Horovod for
                              distributed training on multiple nodes."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
