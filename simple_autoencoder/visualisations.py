import matplotlib.pyplot as plt
import numpy as np


def plot_results_epoch(batch, reconstructions, latent_vec, epoch, step):
    # print(f"shape batch is {batch.shape}")
    # print(f"shape recons is {reconstructions.shape}")
    # change this
    # recon_as_image = np.reshape(reconstructions[0], (469, 11))
    # original_as_image = np.reshape(batch[0], (469, 11))

    fig2, axs2 = plt.subplots(figsize=(4, 8))
    axs2.scatter(range(len(latent_vec[0])), latent_vec[0], s=np.abs(latent_vec[0])*100)
    axs2.set_title('Latent vector View')
    fig2.savefig(f'results/latent_vec_{epoch}_{step}.png')
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
    plt.savefig(f'results/reconstruction_debug_{epoch}_{step}.png')
    plt.close()

def plot_results_before_after_training():
    pass

def plot_losses(train_losses, eval_losses, steps_per_epoch):
    num_steps = len(train_losses)
    indices_eval = list(range(0, num_steps, num_steps//len(eval_losses)))
    _min = min(len(indices_eval), len(eval_losses))

    plt.figure(figsize=(10,10))
    plt.title('Train Losses over steps')
    plt.plot(train_losses, label='train', color='blue')
    plt.plot(indices_eval[:_min], eval_losses[:_min], label='eval', color='orange')
    plt.xlabel('steps')

    # add xticks for epochs
    plt.xticks(np.arange(0, len(train_losses), steps_per_epoch))
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig(f'results/losses.png')
    plt.close()


def plot_latent_space(latent_vectors, categories):
    points = []
    points = np.array(latent_vectors)

    # Creating a scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(x=points[:, 0], y=points[:, 1], s=2.0,
                c=categories, cmap='tab10', alpha=0.9, zorder=2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.show()
    ax.grid(True, color="lightgray", alpha=1.0, zorder=0)

    # Do not show but only save the plot in training
    plt.savefig("results/ls.png", bbox_inches="tight")
    plt.close() # don't forget to close the plot, or it is always in memory

# convert image sequence to a gif file
# def save_gif():

#   frames = []
#   imgs = sorted(os.listdir("./ScatterPlots"))

#   for im in imgs:
#       new_frame = Image.open("./ScatterPlots/" + im)
#       frames.append(new_frame)

#   frames[0].save("latentspace.gif", format="GIF",
#                  append_images=frames[1:],
#                  save_all=True,
#                  duration=200, loop=0)

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
        print(f"shape of original is {originals[i].shape}")
        original = originals[i][:h*w].reshape(h, w) #[:3600].reshape(36, 100)
        reconstruction = reconstructions[i][:h*w].reshape(h, w)
        axs[i, 0].imshow(original, cmap='viridis')
        axs[i, 1].imshow(reconstruction, cmap='viridis')
        axs[i, 2].imshow(np.floor(original*100)/100 - np.floor(reconstruction*100)/100, cmap='gray')

    fig.savefig(f'./results/epoch_{epoch}.png')

def plot_original_reconstruction_fmri(originals, reconstructions, epoch):
    # TODO: implement to show the brain surface with the original and reconstructed fmri data
    pass
