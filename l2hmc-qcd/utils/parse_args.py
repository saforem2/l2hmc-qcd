"""
parse_args.py

Implements method for parsing command line arguments for `gauge_model.py`

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import sys
import argparse
import shlex

import utils.file_io as io

#  from config import process_config
#  from attr_dict import AttrDict

DESCRIPTION = (
    'L2HMC model using U(1) lattice gauge theory for target distribution.'
)


# =============================================================================
#  * NOTE:
#      - if action == 'store_true':
#          The argument is FALSE by default. Passing this flag will cause the
#          argument to be ''stored true''.
#      - if action == 'store_false':
#          The argument is TRUE by default. Passing this flag will cause the
#          argument to be ''stored false''.
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        fromfile_prefix_chars='@',
    )
    ###########################################################################
    #                          Lattice parameters                             #
    ###########################################################################
    parser.add_argument("--space_size",
                        dest="space_size",
                        type=int,
                        default=8,
                        required=False,
                        help="""Spatial extent of lattice.\n (Default: 8)""")

    parser.add_argument("--time_size",
                        dest="time_size",
                        type=int,
                        default=8,
                        required=False,
                        help="""Temporal extent of lattice.\n (Default: 8)""")

    parser.add_argument("--link_type",
                        dest="link_type",
                        type=str,
                        required=False,
                        default='U1',
                        help="""Link type for gauge model.\n
                        (Default: 'U1')""")

    parser.add_argument("--dim",
                        dest="dim",
                        type=int,
                        required=False,
                        default=2,
                        help="""Dimensionality of lattice.\n (Default: 2)""")

    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=128,
                        required=False,
                        help=("""Number of samples (batch size) to use for
                              training.\n (Default: 20)"""))

    parser.add_argument("--rand",
                        dest="rand",
                        action="store_true",
                        required=False,
                        help=("""If passed, set `rand=True` and start lattice
                              from randomized initial configuration.\n
                              (Default: `rand=False`, i.e. NOT passed.)"""))

    ###########################################################################
    #                          Leapfrog parameters                            #
    ###########################################################################

    parser.add_argument("-n", "--num_steps",
                        dest="num_steps",
                        type=int,
                        default=5,
                        required=False,
                        help=("""Number of leapfrog steps to use in (augmented)
                              HMC sampler. (Default: 5)"""))

    parser.add_argument("--eps",
                        dest="eps",
                        type=float,
                        default=0.1,
                        required=False,
                        help=("""Step size to use in leapfrog integrator.
                              (Default: 0.1)"""))

    parser.add_argument("--loss_scale",
                        dest="loss_scale",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Scaling factor to be used in loss function.
                              (lambda in Eq. 7 of paper). (Default: 1.)"""))

    ###########################################################################
    #                       Learning rate parameters                          #
    ###########################################################################

    parser.add_argument("--lr_init",
                        dest="lr_init",
                        type=float,
                        default=1e-3,
                        required=False,
                        help=("""Initial value of learning rate.
                              (Default: 1e-3"""))

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

    ###########################################################################
    #                      Annealing rate parameters                          #
    ###########################################################################

    parser.add_argument("--fixed_beta",
                        dest="fixed_beta",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed runs the training loop
                              at fixed beta (i.e. no annealing is done).
                              (Default: `fixed_beta=False`)"""))

    parser.add_argument("--beta_init",
                        dest="beta_init",
                        type=float,
                        default=2.,
                        required=False,
                        help=("""Initial value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing. (Default: 2.)"""))

    parser.add_argument("--beta_final",
                        dest="beta_final",
                        type=float,
                        default=5.,
                        required=False,
                        help=("""Final value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing. (Default: 5."""))

    parser.add_argument("--warmup_lr",
                        dest="warmup_lr",
                        action="store_true",
                        required=False,
                        help=("""FLag that when passed will 'warmup' the
                              learning rate (i.e. gradually scale it up to the
                              value passed to `--lr_init` (performs better when
                              using Horovod for distributed training)."""))

    ###########################################################################
    #                       Training parameters                               #
    ###########################################################################

    parser.add_argument("--train_steps",
                        dest="train_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of training steps to perform.
                              (Default: 5000)"""))

    parser.add_argument("--trace",
                        dest="trace",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will set `--trace=True`,
                              and create a trace during training loop.
                              (Default: `--trace=False`, i.e.  not passed)"""))

    parser.add_argument("--save_steps",
                        dest="save_steps",
                        type=int,
                        default=50,
                        required=False,
                        help=("""Number of steps after which to save the model
                              and current values of all parameters.
                              (Default: 50)"""))

    parser.add_argument("--print_steps",
                        dest="print_steps",
                        type=int,
                        default=1,
                        required=False,
                        help=("""Number of steps after which to display
                              information about the loss and various
                              other quantities. (Default: 1)"""))

    parser.add_argument("--logging_steps",
                        dest="logging_steps",
                        type=int,
                        default=10,
                        required=False,
                        help=("""Number of steps after which to write logs for
                              tensorboard. (default: 50)"""))

    # ------------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------------

    parser.add_argument('--network_arch',
                        dest='network_arch',
                        type=str,
                        default='conv3D',
                        required=False,
                        help=("""String specifying the architecture to use for
                              the neural network. Must be one of:
                              `'conv3D', 'conv2D', 'generic'`.
                              (Default: 'conv3D')"""))

    parser.add_argument('--num_hidden1',
                        dest='num_hidden1',
                        type=int,
                        default=100,
                        required=False,
                        help=("""Number of nodes to include in each of the
                              fully-connected hidden layers for x, v, and t.
                              (Default: 100)"""))

    parser.add_argument('--num_hidden2',
                        dest='num_hidden2',
                        type=int,
                        default=100,
                        required=False,
                        help=("""Number of nodes to include in fully-connected
                              hidden layer `h`. If not explicitly passed, will
                              default to 2 * lattice.num_links.
                              (Default: None)"""))

    parser.add_argument('--no_summaries',
                        dest="no_summaries",
                        action="store_true",
                        required=False,
                        help=("""FLag that when passed will prevent tensorflow
                              from creating tensorboard summary objects."""))

    parser.add_argument("--hmc",
                        dest="hmc",
                        action="store_true",
                        required=False,
                        help=("""Use generic HMC (without augmented leapfrog
                              integrator described in paper). Used for
                              comparing against L2HMC algorithm."""))

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=10000,
                        required=False,
                        help=("""Number of evaluation 'run' steps to perform
                              after training (i.e. length of desired chain
                              generate using trained L2HMC sample).
                              (Default: 10000)"""))

    parser.add_argument("--eps_fixed",
                        dest="eps_fixed",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will cause the step size
                              `eps` to be a fixed (non-trainable)
                              parameter."""))

    parser.add_argument("--metric",
                        dest="metric",
                        type=str,
                        default="cos_diff",
                        required=False,
                        help=("""Metric to use in loss function. Must be one
                              of: `l1`, `l2`, `cos`, `cos2`, `cos_diff`."""))

    parser.add_argument("--std_weight",
                        dest="std_weight",
                        type=float,
                        default=1,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of stdiliary term in loss function.
                              (Default: 1.)"""))

    parser.add_argument("--aux_weight",
                        dest="aux_weight",
                        type=float,
                        default=1,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of auxiliary term in loss function.
                              (Default: 1.)"""))

    parser.add_argument("--charge_weight",
                        dest="charge_weight",
                        type=float,
                        default=1,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of top. charge term in loss
                              function. (Default: 1.)"""))

    parser.add_argument('--zero_masks',
                        dest='zero_masks',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will set the random
                              binary masks to be all zeros (and its complement
                              to be all ones), instead of both having half of
                              their entries equal to zero and half equal to
                              one."""))

    parser.add_argument('--x_scale_weight',
                        dest='x_scale_weight',
                        type=float,
                        default=1,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--x_translation_weight',
                        dest='x_translation_weight',
                        type=float,
                        default=1,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--x_transformation_weight',
                        dest='x_transformation_weight',
                        type=float,
                        default=1,
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
                        default=1,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--v_translation_weight',
                        dest='v_translation_weight',
                        type=float,
                        default=1,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--v_transformation_weight',
                        dest='v_transformation_weight',
                        type=float,
                        default=1,
                        required=False,
                        help=("""Specify the value of the
                              `transformation_weight` parameter, a
                              multiplicative weight that scales the
                              contribution of the `transformation` (Q) function
                              when performing the augmented L2HMC molecular
                              dynamics update."""))

    parser.add_argument('--use_gaussian_loss',
                        dest='use_gaussian_loss',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will use a `Gaussian`
                              function, exp((x - x0) ** 2 / (2 * sigma)), where
                              `x = metric_fn(x_init, x_proposed) * accept_prob`
                              (i.e. the expected jump distance) is used in the
                              exponential."""))

    parser.add_argument('--use_nnehmc_loss',
                        dest='use_nnehmc_loss',
                        action='store_true',
                        required=False,
                        help=("""If passed, set `use_nnehmc_loss=True` and
                              use alternative NNEHMC loss function."""))

    parser.add_argument("--profiler",
                        dest='profiler',
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will profile the graph
                              execution using `TFProf`."""))

    #  parser.add_argument("--global_seed",
    #                      dest='global_seed',
    #                      type=int,
    #                      default=42,
    #                      required=False,
    #                      help=("""Sets global seed to ensure
    #                            reproducibility."""))

    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed indicates we're training
                              using an NVIDIA GPU."""))

    parser.add_argument("--use_bn",
                        dest='use_bn',
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed causes batch
                              normalization layer to be used in ConvNet."""))

    parser.add_argument("--horovod",
                        dest="horovod",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed uses Horovod for
                              distributed training on multiple nodes."""))

    parser.add_argument("--comet",
                        dest="comet",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed uses comet.ml for
                              parameter logging and additonal metric
                              tracking/displaying."""))

    parser.add_argument("--dropout_prob",
                        dest="dropout_prob",
                        type=float,
                        required=False,
                        default=0.,
                        help=("""Dropout probability in network. If > 0,
                              dropout will be used. (Default: 0.)"""))

    #########################
    #  (Mostly) Deprecated  #
    #########################

    parser.add_argument('--save_samples',
                        dest='save_samples',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will set
                              `--save_samples=True`, and save the samples
                              generated during the `run` phase.
                              (Default: `--save_samples=False, i.e.
                              `--save_samples` is not passed).\n
                              WARNING!! This is very data intensive."""))

    parser.add_argument('--save_lf',
                        dest='save_lf',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will save the
                              output from each leapfrog step."""))

    parser.add_argument("--clip_value",
                        dest="clip_value",
                        type=float,
                        default=0.,
                        required=False,
                        help=("""Clip value, used for clipping value of
                              gradients by global norm. (Default: 0.) If a
                              value greater than 0. is passed, gradient
                              clipping will be performed."""))

    parser.add_argument("--restore",
                        dest="restore",
                        action="store_true",
                        required=False,
                        help=("""Restore model from previous run.  If this
                              argument is passed, a `log_dir` must be specified
                              and passed to `--log_dir argument."""))

    parser.add_argument("--theta",
                        dest="theta",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed indicates we're training
                              on theta @ ALCf."""))

    parser.add_argument("--log_dir",
                        dest="log_dir",
                        type=str,
                        default=None,
                        required=False,
                        help=("""Log directory to use from previous run.
                              If this argument is not passed, a new
                              directory will be created."""))

    parser.add_argument("--num_intra_threads",
                        dest="num_intra_threads",
                        type=int,
                        default=0,
                        required=False,
                        help=("""Number of intra op threads to use for
                              tf.ConfigProto.intra_op_parallelism_threads"""))

    parser.add_argument("--num_inter_threads",
                        dest="num_intra_threads",
                        type=int,
                        default=0,
                        required=False,
                        help=("""Number of intra op threads to use for
                              tf.ConfigProto.intra_op_parallelism_threads"""))

    parser.add_argument("--float64",
                        dest="float64",
                        action="store_true",
                        required=False,
                        help=("""When passed, using 64 point floating precision
                              by settings globals.TF_FLOAT = tf.float64. False
                              by default (use tf.float32)."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
