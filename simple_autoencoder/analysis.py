import sys
from omegacli import OmegaConf, generate_config_template, parse_config
import argparse
import train

def main(argv):
    parser = argparse.ArgumentParser("latent spaces in SAE analysis")
    parser.add_argument("--roi_class", dest='config.roi_class', default='floc-bodies') # floc-bodies, ...
    parser.add_argument("--hem", dest='config.hem', default='lh') # lh, rh, all
    user_provided_args, default_args = OmegaConf.from_argparse(parser)

    train.train_and_evaluate(user_provided_args.config)

def load_model_checkpoint():
    pass


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
