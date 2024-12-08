import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Tuple, List
import jax.numpy as jnp

ds_sizes = {
    'mnist': (28, 28),
    'cifar10': (32, 32),
    'fmri': (100, 32),
}
# training related
def plot_losses(train_losses_mse: List, train_losses_spa: List, results_folder: str, eval_losses: List, steps_per_epoch: int)-> None:
    num_steps = len(train_losses_mse)
    indices_eval = list(range(0, num_steps, num_steps//len(eval_losses)))
    _min = min(len(indices_eval), len(eval_losses))

    plt.figure(figsize=(10,10))
    plt.title('Train Losses over steps')
    plt.plot(train_losses_mse, label='train mse', color='blue')
    # plt.plot(train_losses_spa, label='train spa', color='gray')
    plt.plot(eval_losses, label='eval', color='orange')
    plt.xlabel('steps')

    # add xticks for epochs
    plt.xticks(np.arange(0, len(train_losses_mse), steps_per_epoch))
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig(f'{results_folder}/losses.png')
    plt.close()


def plot_original_reconstruction(originals: jnp.ndarray, reconstructions: jnp.ndarray, config, epoch: int )-> None:
    fig,axs = plt.subplots(5, 3, figsize=(15,15))
    axs[0, 0].set_title('original')
    axs[0, 1].set_title('reconstructed')
    axs[0, 2].set_title('difference')
    # print(originals.shape)

    # plot the first 5 samples of the first batch evaluated
    h, w = ds_sizes[config.ds]
    for i in range(5):
        original = originals[i][:h*w].reshape(h, w) #[:3600].reshape(36, 100)
        reconstruction = reconstructions[i][:h*w].reshape(h, w)
        axs[i, 0].imshow(original, cmap='viridis')
        axs[i, 1].imshow(reconstruction, cmap='viridis')
        axs[i, 2].imshow(np.floor(original*100)/100 - np.floor(reconstruction*100)/100, cmap='gray')

    fig.savefig(f'{config.results_folder}/reconstruction_{epoch}.png')

def plot_original_reconstruction_fmri(originals, reconstructions, results_folder, epoch):
    # TODO: implement to show the brain surface with the original and reconstructed fmri data
    # fig.savefig(f'/{results_folder}/reconstruction_{epoch}.png')
    pass
# %% data analysis
from coco_load import getSubCatjDf, filterByCategory, nsd_coco

def plot_category_distribution(category='person', subjects=[1,2,3,4,5,6,7,8]):

    counts_person, counts_not_person = [], []
    for subj in subjects:
        df = getSubCatjDf(nsd_coco, subj)
        pers = filterByCategory(df, category)
        counts_person.append(pers.shape[0])
        counts_not_person.append(df.shape[0]-pers.shape[0])
    
    x = np.arange(len(subjects))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - bar_width/2, counts_person, bar_width, label='Person', color='indigo')
    bars2 = ax.bar(x + bar_width/2, counts_not_person, bar_width, label='Not Person', color='lightblue')

    ax.set_xlabel('Subjects')
    ax.set_ylabel('Counts')
    ax.set_title('Count of Person vs Not Person for Each Subject')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()
    fig.savefig(f'./results/person_per_subj.png')

# %%
from nsd_data import get_train_test_datasets, get_analysis_datasets

def plot_floc_bodies_values_distribution(split_data, split='train', z_scored=True):
    # Flatten each subject's arrays into a single list for box plot
    flattened_data = [arr.flatten() for arr in split_data]

    # fliers styling
    flierprops = dict(marker='o', markeredgecolor='dimgrey', markersize=2, linestyle='none')

    # make box plot
    fig = plt.figure(figsize=(10, 6))
    boxplot_stats = plt.boxplot(flattened_data, patch_artist=True, boxprops=dict(facecolor="lightblue"), flierprops=flierprops)

    plt.xticks(range(1, len(split_data) + 1), [f'Subj {i+1}' for i in range(len(split_data))])
    plt.title(f"Box plot of 'floc-bodies'-masked {split} data per subject")
    plt.xlabel("Subjects")
    plt.ylabel("Z-scored values")
    plt.show()

    # save figure
    fig.savefig(f'./results/values_distribution_{split}.png')

    # print box plot values
    print("Box Plot Values:")
    for i, data in enumerate(flattened_data):
        # Get whisker low and high
        whisker_low = boxplot_stats['whiskers'][2 * i].get_ydata()[1]
        whisker_high = boxplot_stats['whiskers'][2 * i + 1].get_ydata()[1]
        
        # Get median
        median = boxplot_stats['medians'][i].get_ydata()[0]
        
        # Get Q1 and Q3 from the box PathPatch
        box_coords = boxplot_stats['boxes'][i].get_path().vertices
        q1 = box_coords[0, 1]  # First vertex's y-coordinate
        q3 = box_coords[2, 1]  # Third vertex's y-coordinate
        
        # Get fliers (outliers)
        fliers = boxplot_stats['fliers'][i].get_ydata()
        print(f"Subject {i + 1}:")
        print(f"flattened length: {len(flattened_data[i])}")
        print(f"  Whisker low: {whisker_low}")
        print(f"  Whisker high: {whisker_high}")
        print(f"  Median: {median}")
        print(f"  Q1 (25th percentile): {q1}")
        print(f"  Q3 (75th percentile): {q3}")
        print(f"  Fliers (Outliers) min, max: {jnp.min(fliers), jnp.max(fliers)}")

# 
subjects=[1,2,3,4,5,6,7,8]
trains = list()
vals = list()
# analysis = list()

for subj in subjects:
    train, val = get_train_test_datasets(subj, hem='all')
    trains.append(train)
    vals.append(val)
    print(f'train min, max, avg, len: {jnp.min(train)} {jnp.max(train)} {jnp.average(train)} {len(train)}')
    print(f'val min, max, avg, len: {jnp.min(val)} {jnp.max(val)} {jnp.average(val)} {len(val)}')
    # cat, non = get_analysis_datasets('person', subj)
    # analysis.append(np.concatenate((cat, non), axis=0))

plot_floc_bodies_values_distribution(trains, split='train')
# plot_floc_bodies_values_distribution(vals, split='validation')

# %% latent vector related
def visualize_latent_activations(latent_vecs: jnp.ndarray,
                               images: jnp.ndarray,
                               config,
                               epoch: int | str,
                               num_examples: int = 5) -> None:
    """
    Creates a grid showing images and their corresponding latent activations.

    Args:
        latent_vecs: Array of latent vectors [batch_size, latent_dim]
        images: Original input images [batch_size, height, width]
        num_examples: Number of examples to show
    """
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 4))
    plt.suptitle('Samples with their Latent Representations')

    h, w = ds_sizes[config.ds]
    for i in range(num_examples):
        # Show original image
        img = images[i][:h*w].reshape(h, w)
        # if len(images.shape) == 2:  # If images are flattened
        #     img_size = int(np.sqrt(images.shape[1]))
        #     img = images[i][:3600].reshape(img_size, img_size)
        # else:
        #     img = images[i][:3600]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')

        # Show latent activations as bar plot
        axes[1, i].bar(range(len(latent_vecs[i])), latent_vecs[i])
        axes[1, i].set_ylim([latent_vecs.min(), latent_vecs.max()])
        axes[1, i].set_title(f'Latent Vector {i+1}')

    plt.tight_layout()
    plt.savefig(f'{config.results_folder}/latent_activations_{epoch}.png')

def plot_latent_heatmap(latent_vecs: jnp.ndarray,
                       images: jnp.ndarray,
                       config,
                       epoch: int,
                       num_examples: int = 10) -> None:
    """
    Creates a heatmap of latent activations for multiple examples.

    Args:
        latent_vecs: Array of latent vectors [batch_size, latent_dim]
        images: Original input images [batch_size, height, width]
        num_examples: Number of examples to show
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(latent_vecs[:num_examples].T,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Activation'})
    plt.xlabel('Sample')
    plt.ylabel('Latent Dimension')
    plt.title('Latent Space Activation Patterns')
    plt.savefig(f'{config.results_folder}/latent_heatmap_{epoch}.png')

def track_latent_statistics(latent_vecs) -> Tuple[List[float], List[float]]:
    """
    Compute statistics about latent vector sparsity and activation.

    Args:
        latent_vecs: Array of latent vectors [batch_size, latent_dim]

    Returns:
        sparsity: Percentage of zero/near-zero activations
        mean_activation: Mean absolute activation value
    """
    threshold = 1e-5  # Threshold for considering a value as "zero"
    sparsity = (np.abs(latent_vecs) < threshold).mean() * 100
    mean_activation = np.abs(latent_vecs).mean()

    return sparsity, mean_activation

class LatentVisualizer:
    def __init__(self, results_folder):
        self.sparsity_history = []
        self.activation_history = []
        self.results_folder = results_folder

    def update(self, latent_vecs) -> None:
        """Update tracking statistics during training."""
        sparsity, mean_activation = track_latent_statistics(latent_vecs)
        self.sparsity_history.append(sparsity)
        self.activation_history.append(mean_activation)

    def plot_training_history(self) -> None:
        """Plot the evolution of latent statistics during training."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(self.sparsity_history)
        ax1.set_title('Latent Vector Sparsity Over Training')
        ax1.set_ylabel('% Near-Zero Values')
        ax1.set_xlabel('Training Step')

        ax2.plot(self.activation_history)
        ax2.set_title('Mean Latent Vector Activation Over Training')
        ax2.set_ylabel('Mean Absolute Value')
        ax2.set_xlabel('Training Step')

        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/latent_statistics.png')
