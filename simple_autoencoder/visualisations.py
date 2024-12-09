import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Tuple, List
import jax.numpy as jnp
from surf_plot import plotRoiClassValues, SUBJECTS, plot_img, plotRoiClass

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

def plot_original_reconstruction_fmri(subject:int, originals, reconstructions, hem, roi_class='floc-bodies', style='infl', cmap='cold_hot', total_surface_size=19004+20544):
    """
    Args:
        style (str, optional): ['infl', 'flat', 'sphere']. Defaults to 'infl'.
        total_surface_size (int, otpional): sum of lh fmri size and rh fmri size of the subject. Defaults to 19004 + 20544 (true for most subjects).
    """
    originals = unmask_from_roi_class(subject, originals, roi_class, hem, (originals.shape[0], 19004))
    reconstructions = unmask_from_roi_class(subject, reconstructions, roi_class, hem, (reconstructions.shape[0], 19004))

    originals_lh, originals_rh = split_hemispheres(originals)
    recon_lh, recon_rh = split_hemispheres(reconstructions)

    fig = plt.figure(layout='constrained', figsize=(16, 12))
    fig.suptitle(f'Trained Subject {subject}')
    ogs, recons = fig.subfigures(1, 2, wspace=0.0)
    ogs.suptitle('original')
    recons.suptitle('reconstructed')

    def create_figs(fig):
        lh, rh = fig.subfigures(1, 2, wspace=0.0)
        lh.suptitle('left hemisphere')
        rh.suptitle('right hemisphere')
        return lh.subfigures(3, 1, wspace=0, hspace=0.0), rh.subfigures(3, 1, wspace=0.0, hspace=0.0)

    og_lhs, og_rhs = create_figs(ogs)
    recon_lhs, recon_rhs = create_figs(recons)

    for i in range(3):
        plotRoiClassValues(subject, fmri=originals_lh, img=i, roi_class=roi_class, hemi='lh', cmap=cmap, style=style, fig=og_lhs[i])
        plotRoiClassValues(subject, fmri=originals_rh, img=i, roi_class=roi_class, hemi='rh', cmap=cmap, style=style, fig=og_rhs[i])
        plotRoiClassValues(subject, fmri=recon_lh, img=i, roi_class=roi_class, hemi='lh', cmap=cmap, style=style, fig=recon_lhs[i])
        plotRoiClassValues(subject, fmri=recon_rh, img=i, roi_class=roi_class, hemi='rh', cmap=cmap, style=style, fig=recon_rhs[i])

    fig.savefig(f'./results/subj{subject}_ogs_recons.png', bbox_inches='tight', dpi=150)

# %%
def plot_img_and_fmris(shared_idxs_all, subjects=[1,2,3,4,5], roi_class='floc-bodies'):
    """Args:
        shared_idxs_all (dict): is a dictionary mapping subjects to a tuple of the category and non category image indices
    """
    img_subj=subjects[0]
    img_cat, img_non = shared_idxs_all[img_subj]
    count = len(img_cat) + len(img_non)
    countSub = len(subjects)

    fig = plt.figure(layout='constrained', figsize=(12, 12))
    left, right = fig.subfigures(1, 2, wspace=0.0, width_ratios=[1,2])
    left.suptitle('Stimuli')
    right.suptitle("Lh and rh fMRI responses")

    fmris = right.subfigures(count, 1, wspace=0.0, hspace=0.02)
    imgs = left.subfigures(count, 1, hspace=0.02)

    def create_figs(fig):
        subjs = fig.subfigures(2, countSub, wspace=0.0, hspace=0.0)
        for i in range(countSub): subjs[0, i].suptitle(f"Subj {subjects[i]}")
        return subjs
    subjs = [create_figs(fmri) for fmri in fmris]

    for i, img in enumerate(img_cat):
        imgs[i].suptitle(f"Category: Person")
        plot_img(img_subj, img, imgs[i])
        for j, subId in enumerate(subjects):
            img = shared_idxs_all[subId][0][i]
            plotRoiClassValues(subId, SUBJECTS[subId]['lh_fmri'], img, roi_class, 'lh', cmap='cold_hot', style='sphere', fig=subjs[i][0, j])
            plotRoiClassValues(subId, SUBJECTS[subId]['rh_fmri'], img, roi_class, 'rh', cmap='cold_hot', style='sphere', fig=subjs[i][1, j])

    for i, img in enumerate(img_non):
        imgs[i+len(img_cat)].suptitle(f"No Person")
        plot_img(img_subj, img, imgs[i+len(img_cat)])
        for j, subId in enumerate(subjects):
            img = shared_idxs_all[subId][1][i]
            plotRoiClassValues(subId, SUBJECTS[subId]['lh_fmri'], img, roi_class, 'lh', cmap='cold_hot', style='sphere', fig=subjs[i+2][0, j])
            plotRoiClassValues(subId, SUBJECTS[subId]['rh_fmri'], img, roi_class, 'rh', cmap='cold_hot', style='sphere', fig=subjs[i+2][1, j])

    fig.savefig(f'./results/stimuli-and-fmris-2.png', bbox_inches='tight', dpi=150)

def plot_roi_class_subjs(roi_class: str, subjects: list):
    fig = plt.figure(layout='constrained', figsize=(16, 12))
    subj_figs = fig.subfigures(3, 3, wspace=0.0)

    for i, subjId in enumerate(subjects):
        row = i // 3
        col = i % 3

        subj_fig = subj_figs[row, col]
        subj_fig.suptitle(f'Subj {subjId}, lh and rh')
        lh, rh = subj_fig.subfigures(1,2, wspace=-0.5)
        plotRoiClass(subjId, roi_class, 'lh', cmap='gist_rainbow', style='infl', fig=lh)
        plotRoiClass(subjId, roi_class, 'rh', cmap='gist_rainbow', style='infl', fig=rh)

    fig.savefig(f'./results/subjs-{roi_class}-rois.png', bbox_inches='tight', dpi=150)


# latent vector related
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
