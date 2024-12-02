import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Tuple, List

# training related
def plot_losses(train_losses_mse, train_losses_spa, eval_losses, steps_per_epoch):
    num_steps = len(train_losses_mse)
    indices_eval = list(range(0, num_steps, num_steps//len(eval_losses)))
    _min = min(len(indices_eval), len(eval_losses))

    plt.figure(figsize=(10,10))
    plt.title('Train Losses over steps')
    plt.plot(train_losses_mse, label='train mse', color='blue')
    plt.plot(train_losses_spa, label='train spa', color='gray')
    plt.plot(indices_eval[:_min], eval_losses[:_min], label='eval', color='orange')
    plt.xlabel('steps')

    # add xticks for epochs
    plt.xticks(np.arange(0, len(train_losses_mse), steps_per_epoch))
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig('results/losses.png')
    plt.close()


def plot_original_reconstruction(originals, reconstructions, dataset, epoch ):
    ds_sizes = {
        'mnist': (28, 28),
        'cifar10': (32, 32),
        'fmri': (100, 32),
    }
    fig,axs = plt.subplots(5, 3, figsize=(15,15))
    axs[0, 0].set_title('original')
    axs[0, 1].set_title('reconstructed')
    axs[0, 2].set_title('difference')
    # print(originals.shape)

    # plot the first 5 samples of the first batch evaluated
    h, w = ds_sizes[dataset]
    for i in range(5):
        original = originals[i][:h*w].reshape(h, w) #[:3600].reshape(36, 100)
        reconstruction = reconstructions[i][:h*w].reshape(h, w)
        axs[i, 0].imshow(original, cmap='viridis')
        axs[i, 1].imshow(reconstruction, cmap='viridis')
        axs[i, 2].imshow(np.floor(original*100)/100 - np.floor(reconstruction*100)/100, cmap='gray')

    fig.savefig(f'./results/epoch_{epoch}.png')

def plot_original_reconstruction_fmri(originals, reconstructions, epoch):
    # TODO: implement to show the brain surface with the original and reconstructed fmri data
    pass


# latent vector related

def visualize_latent_activations(latent_vecs,
                               images: np.ndarray,
                                epoch: int,
                               num_examples: int = 5) -> None:
    """
    Creates a grid showing images and their corresponding latent activations.

    Args:
        latent_vecs: Array of latent vectors [batch_size, latent_dim]
        images: Original input images [batch_size, height, width]
        num_examples: Number of examples to show
    """
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 4))
    plt.suptitle('Images and their Latent Representations')

    for i in range(num_examples):
        # Show original image
        if len(images.shape) == 2:  # If images are flattened
            img_size = int(np.sqrt(images.shape[1]))
            img = images[i][:3600].reshape(img_size, img_size)
        else:
            img = images[i][:3600]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')

        # Show latent activations as bar plot
        axes[1, i].bar(range(len(latent_vecs[i])), latent_vecs[i])
        axes[1, i].set_ylim([latent_vecs.min(), latent_vecs.max()])
        axes[1, i].set_title(f'Latent Vector {i+1}')

    plt.tight_layout()
    plt.savefig(f'results/latent_activations_{epoch}.png')

def plot_latent_heatmap(latent_vecs,
                       images: np.ndarray,
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
    plt.xlabel('Example Number')
    plt.ylabel('Latent Dimension')
    plt.title('Latent Space Activation Patterns')
    plt.savefig(f'results/latent_heatmap{epoch}.png')

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
    def __init__(self):
        self.sparsity_history = []
        self.activation_history = []

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
        plt.savefig('results/latent_statistics.png')
