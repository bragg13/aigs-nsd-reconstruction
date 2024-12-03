import sys
from omegacli import OmegaConf, generate_config_template, parse_config
import argparse
import orbax.checkpoint as ocp
from flax.training import train_state
from typing import Any, Tuple
import jax
import jax.numpy as jnp
import models
import ml_collections
import optax
import jaxpruner
from ae_main import PROJECT_DIR
import pathlib

class TrainState(train_state.TrainState):
    batch_stats: Any


# input shape will be (batch_size, fmri_voxels) and batch_size can be set to 1
def load_model_checkpoint(input_shape: Tuple, config, checkpoint_path):
    checkpointer = ocp.StandardCheckpointer()

    # initial dummy data
    rng = jax.random.PRNGKey(0)
    init_key, dropout_key = jax.random.split(rng)
    fmri_voxels = input_shape[1]
    init_data = jnp.ones(input_shape, jnp.float32)

    # initialise the model as in train.py
    model = models.model(config.latent_dim, fmri_voxels, dataset=config.ds)

    variables = model.init(
        {'params': init_key, 'dropout': dropout_key},
        init_data,
        dropout_rng=dropout_key,
        training=False  # we are doing inference
    )

    # configure sparsity as in training TODO: load this somewhere or idk
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = 'static_sparse'
    sparsity_config.sparsity = config.sparsity
    sparsity_config.dist_type = 'erk'

    sparsity_updater = jaxpruner.create_updater_from_config(sparsity_config)
    tx = optax.adamw(config.learning_rate)
    tx = sparsity_updater.wrap_optax(tx)

    # create an initial state
    initial_state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=tx
    )

    # restore the actual state from checkpoint
    restored_state = checkpointer.restore(checkpoint_path / 'state', initial_state)
    return restored_state

def inference(state: TrainState, input_data: jnp.ndarray, config):
    # dummy dropout key
    dropout_key = jax.random.PRNGKey(0)

    variables = {'params': state.params, 'batch_stats': state.batch_stats}

    # would also return the new model state that we dont need
    (reconstructions, latent_vectors), _ = models.model(
        config.latent_dim, input_data.shape[1], dataset=config.ds
    ).apply(
        variables,
        input_data,
        dropout_rng=dropout_key,
        training=False, # we are doing inference
        mutable=['batch_stats']
    )

    return reconstructions, latent_vectors

def save_latent_vectors(latent_vectors, results_folder):
    with open(f"{results_folder}/latent_vectors.npy", 'wb') as f:
        jnp.save(f, latent_vectors)

def main(argv):
    parser = argparse.ArgumentParser("latent spaces in SAE analysis")
    parser.add_argument("--config", dest='config.model_config', default='')
    parser.add_argument("--samples", dest='config.samples', type=int, default=10) # -1 to get all
    parser.add_argument("--subject", dest='config.subject', type=int, default=3) # -1 to get all
    user_provided_args, default_args = OmegaConf.from_argparse(parser)
    subjects = []

    # iterate over all subjects or only one
    if user_provided_args.config.subject == -1:
        subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        subjects.append(user_provided_args.config.subject)

    for subject in subjects:
        # load the model config
        model_config = OmegaConf.load(f"{PROJECT_DIR}/results/{user_provided_args.config.model_config}")

        # load the checkpoint folder
        ckpt_folder = pathlib.Path(f"{PROJECT_DIR}/{user_provided_args.config.results_folder}")

        # load test data
        shared_images = None

        # restore the models
        model = load_model_checkpoint(shared_images.shape, model_config, ckpt_folder)

        # perform some inference
        # reconstructions, latent_vectors = inference(model, shared_images, model_config)

        # visualize the results
        # plot_original_reconstruction(evaluated_batches, reconstructions, config, epoch)
        # visualize_latent_activations(latent_vecs, evaluated_batches, config.results_folder,epoch)
        # plot_latent_heatmap(latent_vecs, evaluated_batches, config.results_folder,epoch)

        # save the latent vectors to disk
        # save_latent_vectors(latent_vectors, model_config.results_folder)

if __name__ == '__main__':
    argv = sys.argv
    main(argv)
