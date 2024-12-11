# %%
import sys
from omegacli import OmegaConf
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
import pathlib
from logger import log
from nsd_data import get_analysis_datasets
from visualisations import (
    visualize_latent_activations,
    plot_original_reconstruction_fmri,
)

PROJECT_DIR = "/Users/andrea/Desktop/aigs/simple_autoencoder/"  # repetition
CHALLENGE_SPACE = {
    "1": (19004, 20544),
    "2": (19004, 20544),
    "3": (19004, 20544),
    "4": (19004, 20544),
    "5": (19004, 20544),
    "6": (18978, 20220),
    "7": (19004, 20544),
    "8": (18981, 20530),
}


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
    model = models.model(config["latent_dim"], fmri_voxels, dataset=config["ds"])

    variables = model.init(
        {"params": init_key, "dropout": dropout_key},
        init_data,
        dropout_rng=dropout_key,
        training=False,  # we are doing inference
    )

    # configure sparsity as in training TODO: load this somewhere or idk
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = "static_sparse"
    sparsity_config.sparsity = config["sparsity"]
    sparsity_config.dist_type = "erk"

    sparsity_updater = jaxpruner.create_updater_from_config(sparsity_config)
    tx = optax.adamw(config["learning_rate"])
    tx = sparsity_updater.wrap_optax(tx)

    # create an initial state
    initial_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )

    # restore the actual state from checkpoint
    restored_state = checkpointer.restore(checkpoint_path / "final", initial_state)
    return restored_state


def inference(state: TrainState, input_data: jnp.ndarray, config):
    def compute_metrics(recon_x, x, latent_vec, config):
        mse_loss = jnp.mean(jnp.square(recon_x - x))
        return mse_loss

    # dummy dropout key
    dropout_key = jax.random.PRNGKey(0)

    variables = {"params": state.params, "batch_stats": state.batch_stats}

    # would also return the new model state that we dont need
    (reconstructions, latent_vectors), _ = models.model(
        config["latent_dim"], input_data.shape[1], dataset=config["ds"]
    ).apply(
        variables,
        input_data,
        dropout_rng=dropout_key,
        training=False,  # we are doing inference
        mutable=["batch_stats"],
    )
    loss = compute_metrics(reconstructions, input_data, latent_vectors, config)
    print(loss)

    return reconstructions, latent_vectors


def save_latent_vectors(latent_vectors, min_num_category, subject, results_folder):
    person = latent_vectors[:min_num_category]
    not_person = latent_vectors[min_num_category:]
    with open(f"{results_folder}/subj0{subject}_shared_person.npy", "wb") as f:
        jnp.save(f, jnp.array(person))
    with open(f"{results_folder}/subj0{subject}_shared_not_person.npy", "wb") as f:
        jnp.save(f, jnp.array(not_person))


def load_model_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f.readlines():
            key, value = line.strip().split(":")
            if key in ["ds", "results_folder", "roi_class", "hem"]:
                config[key] = value
            else:
                config[key] = float(value) if "." in value else int(value)
    return config


def main(argv):
    parser = argparse.ArgumentParser("latent spaces in SAE analysis")
    parser.add_argument(
        "--config", dest="config.model_config", default=""
    )  # mnist_latentXX_sparsityXX_bsXX_lOneXX
    # parser.add_argument("--samples", dest='config.samples', type=int, default=10) # -1 to get all
    # parser.add_argument("--subject", dest='config.subject', type=int, default=3) # -1 to get all
    user_provided_args, default_args = OmegaConf.from_argparse(parser)
    # shared_images = jnp.ones((1000, 784), jnp.float32)
    # _, shared_images = get_train_test_mnist()

    subjects = [1, 5]  # , 4, 5, 6, 7, 8]

    # these are manually inferred from the data
    min_num_person = 382
    min_num_non_person = 382
    # latent_vectors = { f'subj{index}':{'person':[], 'non_person':[]} for index in subjects } # structure to save latent vectors
    # reconstructions = { f'subj{index}':{'person':[], 'non_person':[]} for index in subjects } # structure to save reconstructions (not used rn)

    for subject in subjects:
        # load the model config
        model_config = load_model_config(
            f"{PROJECT_DIR}results/subj{subject}/{user_provided_args.config.model_config}/config"
        )
        log("loaded model config", "ANALYSIS")
        print(model_config)

        # load the analysis data
        # ndr: the images are the same, but I need subject to access the right fmri indices
        person, non_person = get_analysis_datasets(
            "person",
            subject,
            roi_class=model_config["roi_class"],
            hem=model_config["hem"],
        )
        log(f"shared - person - shape: {person.shape}", "ANALYSIS")
        log(f"shared - non person - shape: {non_person.shape}", "ANALYSIS")

        # select the same number of images
        person = person[:min_num_person]
        non_person = non_person[:min_num_non_person]
        shared_images = jnp.concatenate((person, non_person), axis=0)

        # load the checkpoint folder
        ckpt_folder = pathlib.Path(
            f"{PROJECT_DIR}results/subj{subject}/{user_provided_args.config.model_config}/checkpoints"
        )

        # restore the models
        model = load_model_checkpoint(shared_images.shape, model_config, ckpt_folder)
        log("loaded model checkpoint", "ANALYSIS")
        # print(model.params.keys())

        # perform some inference - forse e inutile a sto punto se salvo gli array
        rec, lat_vec = inference(model, shared_images, model_config)
        # latent_vectors[f'subj{subject}']['person'] = lat_vec[:min_num_person]
        # latent_vectors[f'subj{subject}']['non_person'] = lat_vec[min_num_person:]

        # reconstructions[f'subj{subject}']['person'] = rec[:min_num_person]
        # reconstructions[f'subj{subject}']['non_person'] = rec[min_num_person:]

        # visualize the results
        plot_original_reconstruction_fmri(
            subject,
            shared_images,
            rec,
            hem=model_config["hem"],
            lh_chal_space_size=CHALLENGE_SPACE[str(subject)][0],
            rh_chal_space_size=CHALLENGE_SPACE[str(subject)][1],
            roi_class=model_config["roi_class"],
        )
        # plot_original_reconstruction(evaluated_batches, reconstructions, config, epoch)
        # visualize_latent_activations(latent_vectors, shared_images, model_config['results_folder'], 'tested')
        # plot_latent_heatmap(latent_vecs, evaluated_batches, config.results_folder,epoch)

        # save the latent vectors to disk
        # save_latent_vectors(lat_vec, min_num_person, subject, f'{PROJECT_DIR}results/shared')
        del model
        del rec
        del lat_vec
        del person
        del non_person


if __name__ == "__main__":
    argv = sys.argv
    print(argv)
    main(argv)
