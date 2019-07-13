"""
main.py

Complete module capable of training and subsequently running the (trained)
L2HMC sampler on the 2D U(1) lattice gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 07/09/2019
"""
from gauge_model_train import main_training
from gauge_model_inference import main_inference
from utils.parse_args import parse_args


def main(args):
    """Main method for training and subsequently running L2HMC sampler."""
    main_training(args)
    main_inference(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
