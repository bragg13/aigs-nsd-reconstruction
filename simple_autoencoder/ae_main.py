"""Main file for running the AE example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

# from absl import app
# from absl import flags
# from absl import logging
import sys
from clu import platform
import jax
from logger import log
import tensorflow as tf
from omegacli import OmegaConf, generate_config_template, parse_config
import argparse
import train

def main(argv):
    parser = argparse.ArgumentParser("SAE model")
    parser.add_argument("--learning_rate", dest='config.learning_rate', type=float, default=0.0001)
    parser.add_argument("--latent_dim", dest='config.latent_dim', type=int, default=30)
    parser.add_argument("--batch_size", dest='config.batch_size', type=int, default=30)
    parser.add_argument("--num_epochs", dest='config.num_epochs', type=int, default=15)
    parser.add_argument("--roi", dest='config.roi', default='floc-bodies') # floc-bodies, ...
    parser.add_argument("--hem", dest='config.hem', default='lh') # lh, rh, all
    user_provided_args, default_args = OmegaConf.from_argparse(parser)

    log(user_provided_args, "MAIN")
    train.train_and_evaluate(user_provided_args.config)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
