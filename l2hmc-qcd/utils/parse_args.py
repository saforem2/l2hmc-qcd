"""
parse_args.py
Implements method for parsing command line arguments for `gauge_model.py`
Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import json
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

    parser.add_argument('--use_ncp',
                        dest='use_ncp',
                        action='store_true',
                        required=False,
                        help=("""Whether or not to use the NonCompact
                              Projection on the link variables."""))

    parser.add_argument("--compile",
                        dest='compile',
                        action='store_true',
                        required=False,
                        help=("""Whether or not to compile model to graph."""))

    parser.add_argument('--lattice_shape',
                        dest='lattice_shape',
                        type=lambda s: [int(i) for i in s.split(',')],
                        default="128, 16, 16, 2",
                        required=False,
                        help=("""Specifies the shape of our data, with:
                              lattice_shape =
                              (batch_size, time_size, space_size, dim)
                              Defaults to: (128, 16, 16, 2)"""))

    ###########################################################################
    #                          Lattice parameters                             #
    ###########################################################################
    parser.add_argument("--dim",
                        dest="dim",
                        type=int,
                        required=False,
                        default=2,
                        help="""Dimensionality of lattice.\n (Default: 2)""")

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
    parser.add_argument("--beta_init",
                        dest="beta_init",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Initial value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing.\n (Default: 2.)"""))

    parser.add_argument("--beta_final",
                        dest="beta_final",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Final value of beta (inverse coupling
                              constant) used in gauge model when
                              annealing.\n (Default: 5."""))

    parser.add_argument('--warmup_steps',
                        dest='warmup_steps',
                        type=int,
                        required=False,
                        default=0,
                        help=("""Number of steps over which to warmup the
                              learning rate."""))

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

    parser.add_argument("--clip_val",
                        dest="clip_val",
                        type=float,
                        default=0.,
                        required=False,
                        help=("""Clip value, used for clipping value of
                              gradients by global norm. (Default: 0.) If a
                              value greater than 0. is passed, gradient
                              clipping will be performed."""))

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
                        default=0.,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of auxiliary term in loss function.\n
                              (Default: 1.)"""))

    parser.add_argument("--charge_weight",
                        dest="charge_weight",
                        type=float,
                        default=0.1,
                        required=False,
                        help=("""Multiplicative factor used to weigh relative
                              strength of top. charge term in loss
                              function.\n (Default: 0.)"""))

    parser.add_argument("--plaq_weight",
                        dest="plaq_weight",
                        type=float,
                        default=10.,
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

    parser.add_argument('--hmc_steps',
                        dest='hmc_steps',
                        type=int,
                        default=10000,
                        required=False,
                        help=("""Number of steps to train HMC sampler."""))

    parser.add_argument("--log_dir",
                        dest="log_dir",
                        type=str,
                        default=None,
                        required=False,
                        help=("""Log directory to use from previous run.
                              If this argument is not passed, a new
                              directory will be created."""))

    parser.add_argument("--json_file",
                        dest="json_file",
                        type=str,
                        default=None,
                        required=False,
                        help=("""Path to JSON file containing CLI flags.
                              Command line options override values in file.
                              (DEFAULT: None)"""))

    args = parser.parse_args()
    if args.json_file is not None:
        print(f'Loading flags from: {args.json_file}.')
        with open(args.json_file, 'rt') as f:
            t_args = argparse.Namespace()
            # Overwrite parsed args with values from `.json` file
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    return args
