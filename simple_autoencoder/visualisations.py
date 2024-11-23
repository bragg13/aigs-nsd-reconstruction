import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from logger import log

def plot_first_100vx_over_epochs(data):
    # take first 100 voxels from the first batch of each epoch
    print('first 100vx shape')
    print(data.shape)
    data = data[:, 0, :100]
    print(data.shape)

    plt.figure(figsize=(15,15))
    plt.title('First 100 voxels over epochs')
    plt.imshow(data, aspect='auto', cmap='gray')
    plt.savefig('results/first100vx.png')


def plot_results_epoch(batch, reconstructions, latent_vec, epoch):
    # print(f"shape batch is {batch.shape}")
    # print(f"shape recons is {reconstructions.shape}")
    # change this
    recon_as_image = np.reshape(reconstructions[0], (469, 11))
    original_as_image = np.reshape(batch[0], (469, 11))

    fig2, axs2 = plt.subplots(figsize=(4, 8))
    axs2.scatter(range(len(latent_vec[0])), latent_vec[0], s=np.abs(latent_vec[0])*100)
    axs2.set_title('Latent vector View')
    fig2.savefig(f'results/latent_vec_{epoch}.png')
    plt.close()

    # Visualization
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.title('Original Data')
    plt.imshow(batch, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(132)
    plt.title('Reconstructed Data')
    plt.imshow(reconstructions, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(133)
    plt.title('Difference')
    plt.imshow(batch - reconstructions, aspect='auto', cmap='RdBu_r')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'results/reconstruction_debug_{epoch}.png')
    plt.close()

def plot_results_before_after_training():
    pass

def plot_losses(losses):
    log(losses, 'LOSSES')
    plt.figure(figsize=(10,10))
    plt.title('Losses over epochs')
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(f'results/losses.png')
    plt.close()

def plot_data_distribution(lh_fmri, rh_fmri, train: bool):
    fig, axs = plt.subplots(3, figsize=(15,15))
    if lh_fmri is not None: sns.histplot(lh_fmri.reshape(-1), ax=axs[0], kde=True)
    if rh_fmri is not None: sns.histplot(rh_fmri.reshape(-1), ax=axs[1], kde=True)
    axs[0].set_title('Distribution of fMRI Data (lh)')
    axs[0].set_xlabel('Voxel Values')
    axs[0].set_ylabel('Frequency')
    axs[1].set_title('Distribution of fMRI Data (rh)')
    axs[1].set_xlabel('Voxel Values')
    axs[1].set_ylabel('Frequency')

    # Plot overlapping distributions on third axis
    if lh_fmri is not None and rh_fmri is not None:
        # Create normal distribution with same mean/std as data
        x = np.linspace(min(lh_fmri.min(), rh_fmri.min()), max(lh_fmri.max(), rh_fmri.max()), 100)
        mu = np.mean([lh_fmri.mean(), rh_fmri.mean()])
        sigma = np.mean([lh_fmri.std(), rh_fmri.std()])
        normal = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

        sns.histplot(data=lh_fmri.reshape(-1), ax=axs[2], stat='density', color='blue', alpha=0.3, label='LH fMRI')
        sns.histplot(data=rh_fmri.reshape(-1), ax=axs[2], stat='density', color='red', alpha=0.3, label='RH fMRI')
        axs[2].plot(x, normal, color='green', label='Normal Distribution')
        axs[2].legend()
        # TODO: problema, la media e mu sono settati sulla medai e il mu delle distribuzioni, dpovrebber essre sullo zero forse?
        axs[2].set_title('Overlapping Distributions')
        axs[2].set_xlabel('Voxel Values')
        axs[2].set_ylabel('Density')

        print(f"(lh) Mean: {np.mean(lh_fmri)}")
        print(f"(rh) Mean: {np.mean(rh_fmri)}")
        print(f"(lh) std: {np.std(lh_fmri)}")
        print(f"(rh) std: {np.std(rh_fmri)}")

        _type = 'train' if train else 'test'
        fig.savefig(f'results/{_type}_data_distribution.png')
        plt.close()
