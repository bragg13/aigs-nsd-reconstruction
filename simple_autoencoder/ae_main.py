"""Main file for running the AE example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

# from absl import app
# from absl import flags
# from absl import logging
import sys
import tensorflow as tf
from omegacli import OmegaConf
import argparse
import train
import os
PROJECT_DIR = '/Users/andrea/Desktop/aigs/simple_autoencoder/'

def main(argv):
    # hide memory from tensorflow or it miht conflict with jax
    tf.config.experimental.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser("SAE model")
    parser.add_argument("--learning_rate", dest='config.learning_rate', type=float, default=0.0001)
    parser.add_argument("--latent_dim", dest='config.latent_dim', type=int, default=30)
    parser.add_argument("--batch_size", dest='config.batch_size', type=int, default=30)
    parser.add_argument("--num_epochs", dest='config.num_epochs', type=int, default=15)
    parser.add_argument("--roi_class", dest='config.roi_class', default='floc-bodies') # floc-bodies, ...
    parser.add_argument("--hem", dest='config.hem', default='lh') # lh, rh, all
    parser.add_argument("--ds", dest='config.ds', default='fmri') # mnist, cifar10, fmri
    parser.add_argument("--sparsity", dest='config.sparsity', type=float, default=0.8)
    parser.add_argument("--l1", dest='config.l1', type=float, default=0.1)
    parser.add_argument("--subject", dest='config.subject', type=int, default=3)

    user_provided_args, default_args = OmegaConf.from_argparse(parser)

    # create the results folder
    results_folder = f'results/subj{user_provided_args.config.subject}/{user_provided_args.config.ds}_latent{user_provided_args.config.latent_dim}_sparsity{user_provided_args.config.sparsity}_bs{user_provided_args.config.batch_size}_lOne{user_provided_args.config.l1}'
    os.makedirs(results_folder, exist_ok=True)
    user_provided_args.config['results_folder'] = results_folder

    # write the config to the results folder
    OmegaConf.save(user_provided_args.config, os.path.join(results_folder, 'config.yaml'))

    train.train_and_evaluate(user_provided_args.config)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
