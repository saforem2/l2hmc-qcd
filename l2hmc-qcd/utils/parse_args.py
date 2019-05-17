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


def get_args():
    argparser = argparse.ArgumentParser(description=DESCRIPTION)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The configuration file'
    )
    args = argparser.parse_args()

    return args


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

    # ------------------------------------------------------------------------
    # Lattice parameters 
    # ------------------------------------------------------------------------

    parser.add_argument("--space_size",
                        dest="space_size",
                        type=int,
                        default=8,
                        required=False,
                        help="Spatial extent of lattice.")

    parser.add_argument("--time_size",
                        dest="time_size",
                        type=int,
                        default=8,
                        required=False,
                        help="Temporal extent of lattice.")

    parser.add_argument("--link_type",
                        dest="link_type",
                        type=str,
                        required=False,
                        default='U1',
                        help="Link type for gauge model.")

    parser.add_argument("--dim",
                        dest="dim",
                        type=int,
                        required=False,
                        default=2,
                        help="Dimensionality of lattice.")

    parser.add_argument("--num_samples",
                        dest="num_samples",
                        type=int,
                        default=10,
                        required=False,
                        help=("Number of samples (batch size) to use for "
                              "training."))

    parser.add_argument("--rand",
                        dest="rand",
                        action="store_true",
                        required=False,
                        help=("Start lattice from randomized initial "
                              "configuration."))

    # ------------------------------------------------------------------------
    # Leapfrog parameters
    # ------------------------------------------------------------------------

    parser.add_argument("-n", "--num_steps",
                        dest="num_steps",
                        type=int,
                        default=5,
                        required=False,
                        help=("Number of leapfrog steps to use in (augmented) "
                              "HMC sampler."))

    parser.add_argument("--eps",
                        dest="eps",
                        type=float,
                        default=0.1,
                        required=False,
                        help="Step size to use in leapfrog integrator.")

    parser.add_argument("--loss_scale",
                        dest="loss_scale",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Scaling factor to be used in loss function. "
                              "(lambda in Eq. 7 of paper)."))

    # ------------------------------------------------------------------------
    # Learning rate parameters
    # ------------------------------------------------------------------------

    parser.add_argument("--lr_init",
                        dest="lr_init",
                        type=float,
                        default=1e-3,
                        required=False,
                        help="Initial value of learning rate.")

    parser.add_argument("--lr_decay_steps",
                        dest="lr_decay_steps",
                        type=int, default=500,
                        required=False,
                        help=("Number of steps after which to decay learning "
                              "rate."))

    parser.add_argument("--lr_decay_rate",
                        dest="lr_decay_rate",
                        type=float, default=0.96,
                        required=False,
                        help=("Learning rate decay rate to be used during "
                              "training."))

    # ------------------------------------------------------------------------
    # Annealing rate parameters
    # ------------------------------------------------------------------------

    parser.add_argument("--annealing",
                        dest="annealing",
                        action="store_true",
                        required=False,
                        help=("""Flag that when passed will cause the model
                              to perform simulated annealing during
                              training."""))

    parser.add_argument("--hmc_beta",
                        dest="hmc_beta",
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run the generic HMC sampler."""))

    parser.add_argument("--hmc_eps",
                        dest="hmc_eps",
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular step size value
                              (`eps`) to use when running the generic HMC
                              sampler."""))

    parser.add_argument("--beta_init",
                        dest="beta_init",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Initial value of beta (inverse coupling "
                              "constant) used in gauge model when annealing."))

    parser.add_argument("--beta_final",
                        dest="beta_final",
                        type=float,
                        default=8.,
                        required=False,
                        help=("Final value of beta (inverse coupling "
                              "constant) used in gauge model when annealing."))

    # ------------------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------------------

    parser.add_argument("--train_steps",
                        dest="train_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help="Number of training steps to perform.")

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=50000,
                        required=False,
                        help=("Number of evaluation 'run' steps to perform "
                              "after training (i.e. length of desired chain "
                              "generate using trained L2HMC sampler.)"))

    parser.add_argument("--trace",
                        dest="trace",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed will create trace during "
                              "training loop."))

    parser.add_argument("--save_steps",
                        dest="save_steps",
                        type=int,
                        default=50,
                        required=False,
                        help=("Number of steps after which to save the model "
                              "and current values of all parameters."))

    parser.add_argument("--print_steps",
                        dest="print_steps",
                        type=int,
                        default=1,
                        required=False,
                        help=("Number of steps after which to display "
                              "information about the loss and various "
                              "other quantities."))

    parser.add_argument("--logging_steps",
                        dest="logging_steps",
                        type=int,
                        default=50,
                        required=False,
                        help=("Number of steps after which to write logs for "
                              "tensorboard."))

    parser.add_argument("--training_samples_steps",
                        dest="training_samples_steps",
                        type=int,
                        default=1000,
                        required=False,
                        help=("Number of intermittent steps after which "
                              "the sampler is evaluated at `beta_final`. "
                              "This allows us to monitor the performance of "
                              "the sampler during training."))

    parser.add_argument("--training_samples_length",
                        dest="training_samples_length",
                        type=int,
                        default=500,
                        required=False,
                        help=("Number of steps to run sampler for when "
                              "evaluating the sampler during training."))

    # ------------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------------

    parser.add_argument('--network_arch',
                        dest='network_arch',
                        type=str,
                        default='conv3D',
                        required=False,
                        help=("String specifying the architecture to use for "
                              "the neural network. Must be one of: "
                              "`'conv3D', 'conv2D', 'generic'`."))

    parser.add_argument('--summaries',
                        dest="summaries",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed creates "
                              "summaries of gradients and variables for "
                              "monitoring in tensorboard."))

    parser.add_argument('--save_samples',
                        dest='save_samples',
                        action='store_true',
                        required=False,
                        help=("Flag that when passed will save the samples "
                              "generated during the `run` phase. "
                              "WARNING: This is very data intensive."))

    parser.add_argument('--save_leapfrogs',
                        dest='save_leapfrogs',
                        action='store_true',
                        required=False,
                        help=("Flag that when passed will save the "
                              "output from each leapfrog step"))

    parser.add_argument('--long_run',
                        dest='long_run',
                        action='store_true',
                        required=False,
                        help=("Flag that when passed runs the trained sampler "
                              "at model.beta_final and model.beta_final + 1."))

    parser.add_argument("--hmc",
                        dest="hmc",
                        action="store_true",
                        required=False,
                        help=("Use generic HMC (without augmented leapfrog "
                              "integrator described in paper). Used for "
                              "comparing against L2HMC algorithm."))

    parser.add_argument("--run_hmc",
                        dest="run_hmc",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed causes generic HMC "
                              "to be ran after running the trained L2HMC "
                              "sampler. (Default: False)"))

    parser.add_argument("--eps_trainable",
                        dest="eps_trainable",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed will allow the step size "
                              "`eps` to be a trainable parameter."))

    parser.add_argument("--metric",
                        dest="metric",
                        type=str,
                        default="cos_diff",
                        required=False,
                        help=("Metric to use in loss function. Must be one "
                              "of: `l1`, `l2`, `cos`, `cos2`, `cos_diff`."))

    parser.add_argument("--std_weight",
                        dest="std_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Multiplicative factor used to weigh relative "
                              "strength of stdiliary term in loss function."))

    parser.add_argument("--aux_weight",
                        dest="aux_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Multiplicative factor used to weigh relative "
                              "strength of auxiliary term in loss function."))

    parser.add_argument("--charge_weight",
                        dest="charge_weight",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Multiplicative factor used to weigh relative "
                              "strength of top. charge term in loss function"))

    parser.add_argument("--clip_grads",
                        dest="clip_grads",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed will clip gradients by "
                              "global norm using `--clip_value` command line "
                              "argument. If `--clip_value` is not passed, "
                              "it defaults to 100."))

    parser.add_argument("--clip_value",
                        dest="clip_value",
                        type=float,
                        default=1.,
                        required=False,
                        help=("Clip value, used for clipping value of "
                              "gradients by global norm."))

    parser.add_argument("--log_dir",
                        dest="log_dir",
                        type=str,
                        default=None,
                        required=False,
                        help=("Log directory to use from previous run. "
                              "If this argument is not passed, a new "
                              "directory will be created."))

    parser.add_argument("--restore",
                        dest="restore",
                        action="store_true",
                        required=False,
                        help=("Restore model from previous run. "
                              "If this argument is passed, a `log_dir` "
                              "must be specified and passed to `--log_dir` "
                              "argument."))

    parser.add_argument("--profiler",
                        dest='profiler',
                        action="store_true",
                        required=False,
                        help=("Flag that when passed will profile the graph "
                              "execution using `TFProf`."))

    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed indicates we're training "
                              "using an NVIDIA GPU."))

    parser.add_argument("--theta",
                        dest="theta",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed indicates we're training "
                              "on theta @ ALCf."))

    parser.add_argument("--use_bn",
                        dest='use_bn',
                        action="store_true",
                        required=False,
                        help=("Flag that when passed causes batch "
                              "normalization layer to be used in ConvNet"))

    parser.add_argument("--horovod",
                        dest="horovod",
                        action="store_true",
                        required=False,
                        help=("Flag that when passed uses Horovod for "
                              "distributed training on multiple nodes."))

    parser.add_argument("--num_intra_threads",
                        dest="num_intra_threads",
                        type=int,
                        default=0,
                        required=False,
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    parser.add_argument("--num_inter_threads",
                        dest="num_intra_threads",
                        type=int,
                        default=0,
                        required=False,
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    if sys.argv[1].startswith('@'):
        if sys.argv[1].endswith('.json'):
            try:
                args_ = get_args()
                args = process_config(args_.config)
            except:
                io.log("Missing or invalid arguments")
                exit(0)
        else:
            args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                                 comments=True))
    else:
        args = parser.parse_args()

    return args
