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

    parser.add_argument("--num_samples",
                        dest="num_samples",
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO:
    # -------------------------------------------------------------------------
    #   Since we want `--annealing` to be True by default, its annoying to
    #   always explicitly pass `--annealing` as a command line argument to
    #   get this expected behavior.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    parser.add_argument("--hmc_beta",
                        dest="hmc_beta",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run the generic HMC sampler.
                              (Default: None)"""))

    parser.add_argument("--hmc_eps",
                        dest="hmc_eps",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular step size value
                              (`eps`) to use when running the generic HMC
                              sampler. (Default: None)"""))

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

    parser.add_argument('--inference',
                        dest="inference",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will run inference using
                              the trained L2HMC sampler by loading the trained
                              model."""))

    parser.add_argument("--warmup_lr",
                        dest="warmup_lr",
                        action="store_true",
                        required=False,
                        help=("""FLag that when passed will 'warmup' the
                              learning rate (i.e. gradually scale it up to the
                              value passed to `--lr_init` (performs better when
                              using Horovod for distributed training)."""))

    parser.add_argument("--beta_inference",
                        dest="beta_inference",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run inference using the trained
                              L2HMC sampler. (Default: None"""))

    parser.add_argument("--charge_weight_inference",
                        dest="charge_weight_inference",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of the charge
                              weight at which to run inference using the
                              trained L2HMC sampler. (Default: None"""))

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

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=10000,
                        required=False,
                        help=("""Number of evaluation 'run' steps to perform
                              after training (i.e. length of desired chain
                              generate using trained L2HMC sample).
                              (Default: 10000)"""))

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

    parser.add_argument('--num_hidden',
                        dest='num_hidden',
                        type=int,
                        default=100,
                        required=False,
                        help=("""Number of nodes to include in fully-connected
                              hidden layer `h`. If not explicitly passed, will
                              default to 2 * lattice.num_links.
                              (Default: None)"""))

    parser.add_argument('--summaries',
                        dest="summaries",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed sets `--summaries=True`,
                              and creates summaries of gradients and vaiables
                              for monitoring in tensorboard.
                              (Default: `--summaries=False, i.e. `--summaries`
                              is not passed.)"""))

    parser.add_argument('--plot_lf',
                        dest='plot_lf',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will set
                              `--plot_lf=True`, and will plot the 'metric'
                              distance between subsequent configurations, as
                              well as the determinant of the Jacobian of the
                              transformation for each individual leapfrog step,
                              as well as each molecular dynamics step (with
                              Metrpolis-Hastings accept/reject).\n
                              When plotting the determinant of the Jacobian
                              following the MD update, we actually calculate
                              the sum of the determinants from each individual
                              LF step since this is the quantity that actually
                              enters into the MH acceptance probability.
                              (Default: `--plot_lf=False`, i.e. `--plot_lf` is
                              not passed mostly just beause the plots are
                              extremely large (many LF steps during inference)
                              and take a while to actually generate.)"""))

    parser.add_argument('--loop_net_weights',
                        dest='loop_net_weights',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed sets
                              `--loop_net_weights=True`, and will iterate over
                              multiple values of `net_weights`, which are
                              multiplicative scaling factors applied to each of
                              the Q, S, T functions when running the trained
                              sampler.
                              (Default: `--loop_net_weights=False, i.e.
                              `--loop_net_weights is not passed)"""))

    parser.add_argument("--hmc",
                        dest="hmc",
                        action="store_true",
                        required=False,
                        help=("""Use generic HMC (without augmented leapfrog
                              integrator described in paper). Used for
                              comparing against L2HMC algorithm."""))

    parser.add_argument("--run_hmc",
                        dest="run_hmc",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed causes generic HMC
                              to be ran after running the trained L2HMC
                              sampler. (Default: False)"""))

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

    parser.add_argument("--nnehmc_loss",
                        dest="nnehmc_loss",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will calculate the
                              'NNEHMC Loss' from
                              (https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf)
                              (Default: False)."""))

    parser.add_argument("--std_weight",
                        dest="std_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of stdiliary term in loss function.
                              (Default: 1.)"""))

    parser.add_argument("--aux_weight",
                        dest="aux_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of auxiliary term in loss function.
                              (Default: 1.)"""))

    parser.add_argument("--charge_weight",
                        dest="charge_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of top. charge term in loss
                              function. (Default: 1.)"""))

    parser.add_argument("--profiler",
                        dest='profiler',
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will profile the graph
                              execution using `TFProf`."""))

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
