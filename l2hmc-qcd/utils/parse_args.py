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

    parser.add_argument("--separate_networks",
                        dest='separate_networks',
                        action='store_true',
                        required=False,
                        help=("""Whether or not to use separate networks for
                              each MC step."""))

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
                        default=64,
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
                        default=1,
                        required=False,
                        help=("""Number of leapfrog steps to use in (augmented)
                              HMC sampler.\n(Default: 5)"""))

    parser.add_argument("--eps",
                        dest="eps",
                        type=float,
                        default=0.2,
                        required=False,
                        help=("""Step size to use in leapfrog integrator.\n
                              (Default: 0.1)"""))

    parser.add_argument("--loss_scale",
                        dest="loss_scale",
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Scaling factor to be used in loss function.
                              (lambda in Eq. 7 of paper).\n (Default: 1.)"""))

    ###########################################################################
    #                       Learning rate parameters                          #
    ###########################################################################

    parser.add_argument("--lr_init",
                        dest="lr_init",
                        type=float,
                        default=1e-3,
                        required=False,
                        help=("""Initial value of learning rate.\n
                              (Default: 1e-3"""))

    parser.add_argument("--lr_decay_steps",
                        dest="lr_decay_steps",
                        type=int,
                        default=2500,
                        required=False,
                        help=("""Number of steps after which to decay learning
                              rate.\n (Default: 500)"""))

    parser.add_argument("--lr_decay_rate",
                        dest="lr_decay_rate",
                        type=float, default=0.96,
                        required=False,
                        help=("""Learning rate decay rate to be used during
                              training.\n (Default: 0.96)"""))

    ###########################################################################
    #                      Annealing rate parameters                          #
    ###########################################################################

    parser.add_argument("--fixed_beta",
                        dest="fixed_beta",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed runs the training loop
                              at fixed beta (i.e. no annealing is done).\n
                              (Default: `fixed_beta=False`)"""))

    parser.add_argument("--beta_init",
                        dest="beta_init",
                        type=float,
                        default=2.,
                        required=False,
                        help=("""Initial value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing.\n (Default: 2.)"""))

    parser.add_argument("--beta_final",
                        dest="beta_final",
                        type=float,
                        default=5.,
                        required=False,
                        help=("""Final value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing.\n (Default: 5."""))

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
                        default=10000,
                        required=False,
                        help=("""Number of training steps to perform.\n
                              (Default: 10000)"""))

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of inference steps to perform.\n
                              (Default: 5000)"""))

    parser.add_argument('--inference',
                        dest='inference',
                        action='store_true',
                        required=False,
                        help=("""FLag that when passed will run inference on
                              trained model."""))

    parser.add_argument("--extra_steps",
                        dest="extra_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of additional steps to append at the
                              end of the training instance at the final value
                              of beta."""))

    parser.add_argument("--trace",
                        dest="trace",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will set `--trace=True`,
                              and create a trace during training loop.\n
                              (Default: `--trace=False`, i.e.  not passed)"""))

    parser.add_argument("--save_steps",
                        dest="save_steps",
                        type=int,
                        default=50,
                        required=False,
                        help=("""Number of steps after which to save the model
                              and current values of all parameters.\n
                              (Default: 50)"""))

    parser.add_argument("--print_steps",
                        dest="print_steps",
                        type=int,
                        default=1,
                        required=False,
                        help=("""Number of steps after which to display
                              information about the loss and various
                              other quantities.\n (Default: 1)"""))

    parser.add_argument("--logging_steps",
                        dest="logging_steps",
                        type=int,
                        default=100,
                        required=False,
                        help=("""Number of steps after which to write logs for
                              tensorboard.\n (Default: 100)"""))

    # ------------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------------

    parser.add_argument('--network_arch',
                        dest='network_arch',
                        type=str,
                        default='generic',
                        required=False,
                        help=("""String specifying the architecture to use for
                              the neural network. Must be one of:
                              `'conv3D', 'conv2D', 'generic'`.\n
                              (Default: 'conv3D')"""))

    parser.add_argument('--network_type',
                        dest='network_type',
                        type=str,
                        default='GaugeNetwork',
                        required=False,
                        help=("""String specifying the type of network to
                              use. Possible values: `'CartesianNet'`. If not
                              specified, will use generic `'FullNet'`."""))

    parser.add_argument('--units',
                        dest='units',
                        type=lambda s: [int(i) for i in s.split(',')],
                        default="64,128",
                        required=False,
                        help=("""Number of nodes to use in hidden layers. The
                              number of hidden layers will be determined by the
                              number of entries provided. Example: `--units
                              '100,200,300'` will make 3 hidden layers with
                              100, 200, and 300 hidden units respectively.
                              (Default: '64,128')."""))

    parser.add_argument("--dropout_prob",
                        dest="dropout_prob",
                        type=float,
                        required=False,
                        default=0.,
                        help=("""Dropout probability in network. If > 0,
                              dropout will be used. (Default: 0.)"""))

    parser.add_argument("--clip_value",
                        dest="clip_value",
                        type=float,
                        default=0.,
                        required=False,
                        help=("""Clip value, used for clipping value of
                              gradients by global norm. (Default: 0.) If a
                              value greater than 0. is passed, gradient
                              clipping will be performed."""))

    parser.add_argument('--largest_wilson_loop',
                        dest='largest_wilson_loop',
                        type=int,
                        required=False,
                        default=1,
                        help=("""Size of largest Wilson loop to include when
                              calculating the plaquette and charge terms in the
                              gauge loss function."""))

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
                              strength of stdiliary term in loss function.\n
                              (Default: 1.)"""))

    parser.add_argument("--aux_weight",
                        dest="aux_weight",
                        type=float,
                        default=1,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of auxiliary term in loss function.\n
                              (Default: 1.)"""))

    parser.add_argument("--charge_weight",
                        dest="charge_weight",
                        type=float,
                        default=0.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of top. charge term in loss
                              function.\n (Default: 0.)"""))

    parser.add_argument("--plaq_weight",
                        dest="plaq_weight",
                        type=float,
                        default=0.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of plaquette difference term in loss
                              function.\n (Default: 0.)"""))

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

    parser.add_argument("--activation",
                        dest='activation',
                        type=str,
                        default='relu',
                        required=False,
                        help=("""Flag used to specify the activation function
                              to be used in the network."""))

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

    parser.add_argument('--save_train_data',
                        dest='save_train_data',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will save training
                              data."""))

    parser.add_argument("--restore",
                        dest="restore",
                        action="store_true",
                        required=False,
                        help=("""Restore model from previous run.  If this
                              argument is passed, a `log_dir` must be specified
                              and passed to `--log_dir argument."""))

    parser.add_argument('--hmc_start',
                        dest='hmc_start',
                        action='store_true',
                        required=False,
                        help=("""Find optimal `eps` by training HMC model with
                              `eps` a trainable parameter, and use this value
                              along with the resulting thermalized config as
                              the starting point for training the L2HMC
                              sampler."""))

    parser.add_argument('--hmc_steps',
                        dest='hmc_steps',
                        type=int,
                        default=10000,
                        required=False,
                        help=("""Number of steps to train HMC sampler."""))

    parser.add_argument("--resume_training",
                        dest="resume_training",
                        action="store_true",
                        required=False,
                        help=("""Resume training."""))

    parser.add_argument('--to_restore',
                        dest='to_restore',
                        type=lambda s: [str(i) for i in s.split(',')],
                        default='',
                        #  default="x,eps,beta,lr",
                        required=False,
                        help=("""List of variable names to restore if restoring
                              from previous training run.
                              Possible values: ['x', 'beta', 'eps', 'lr']."""))

    parser.add_argument("--theta",
                        dest="theta",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed indicates we're training
                              on theta @ ALCf."""))

    parser.add_argument('--root_dir',
                        dest='root_dir',
                        default='gauge_logs',
                        required=False,
                        help=("""Root directory in which to store data."""))

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

    parser.add_argument("--restart_beta",
                        dest="restart_beta",
                        type=float,
                        default=-1.,
                        required=False,
                        help=("""When restarting training, this will be the new
                               starting value of beta if `--restart_beta >
                              0` (Default: -1)."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
