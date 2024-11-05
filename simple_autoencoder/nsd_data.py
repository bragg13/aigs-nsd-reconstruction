# %%
# import libraries
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import jax_dataloader as jdl
import jax.numpy as jnp
from tqdm.gui import tqdm

# %% class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+ self.subj)

def get_dir(folder: str, subject=3):
    data_dir = '../data'
    args = argObj(data_dir, subject)
    return os.path.join(args.data_dir, 'training_split', folder)

# rand_seed = np.random.seed(5) #@param # from colab tutorial
rand_seed = jdl.manual_seed(1234) # from jdl documentation

# %% Create the training, validation and test partitions indices
# THIS SHOULD INSTEAD RETURN THE INDEXES OF SPECIFIC AND SHARED IMAGES IN THE SUBJ FOLDER
def shuffle_idxs():
    def make_stim_list(stim_dir):
        # Create lists will all training and test image file names, sorted
        stim_list = os.listdir(stim_dir)
        stim_list.sort()
        print('Total stimulus images: ' + str(len(stim_list)))
        return stim_list

    stim_list = make_stim_list(get_dir('training_images'))

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(stim_list) / 100 * 90))
    # Shuffle all training stimulus images
    idxs = np.arange(len(stim_list))
    np.random.shuffle(idxs)

    # Assign 90% of the shuffled stimulus images to the training partition, and 10% to the test partition
    idxs_train, idxs_test = idxs[:num_train], idxs[num_train:]

    # print('Training stimulus images: ' + format(len(idxs_train)))
    # print('Test stimulus images: ' + format(len(idxs_test)))
    print(f'idx of first train image: {idxs_train[0]}')
    print(f'idx of first test image: {idxs_test[0]}')
    return idxs_train, idxs_test

# %%
import jax
def custom_transforms(image):
    # Resize to 224x224
    image = image.resize((224, 224))

    # convert and normalize
    image_array = jnp.array(image) / 255.0
    image_array = (image_array - jnp.array([0.485, 0.456, 0.406])) / jnp.array([0.229, 0.224, 0.225])  # Normalize

    # Rearrange dimensions to match JAX's convention (HWC)
    image_array = jnp.transpose(image_array, (2, 0, 1))  # Change to (C, H, W) if needed for your model
    return image_array

class ImageAndFmriDataset(jdl.Dataset):
    """ Dataset class for loading images and fMRI data """
    def __init__(self, imgs, fmri_paths, idxs):
        self.imgs = imgs[idxs]
        self.lh_fmri = jnp.load(fmri_paths[0])[idxs]
        self.rh_fmri = jnp.load(fmri_paths[1])[idxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # return a tuple containing the image and the fMRI data
        img = self.imgs[idx]
        return img, self.lh_fmri[idx], self.rh_fmri[idx]

# %% main
def create_loaders(all_idxs, batch_size, subject=3):
    # directories - we use shared images for testing, so everything is in the same directory
    imgs_dir = get_dir('training_images', subject)
    imgs_paths = sorted(list(Path(imgs_dir).iterdir())) # could pass directly this

    fmri_dir = get_dir('training_fmri', subject)
    lh_fmri_path = os.path.join(fmri_dir, 'lh_training_fmri.npy')
    rh_fmri_path = os.path.join(fmri_dir, 'rh_training_fmri.npy')
    fmri_paths = [lh_fmri_path, rh_fmri_path]

    imgs = []
    for i in range(len(imgs_paths)):
        img = custom_transforms(Image.open(imgs_paths[i]).convert('RGB'))
        if i % 200 == 0:
            print(i)
        imgs.append(img)
    imgs = jnp.array(imgs)

    # indexes - what actually changes
    idxs_train, idxs_test = all_idxs

    train_dataset = ImageAndFmriDataset(imgs, fmri_paths, idxs_train)
    test_dataset = ImageAndFmriDataset(imgs, fmri_paths, idxs_test)

    train_loader = jdl.DataLoader(
        dataset=train_dataset,
        backend='jax',
        batch_size=batch_size,
        shuffle=True, # now we can shuffle cause we have tuples
        drop_last=False,
    )
    test_loader = jdl.DataLoader(
        dataset=test_dataset,
        backend='jax',
        batch_size=batch_size,
        shuffle=True, # now we can shuffle cause we have tuples
        drop_last=False,
    )

    return train_loader, test_loader


# %% main
idxs_train, idxs_test = shuffle_idxs()
train_loader, test_loader = create_loaders((idxs_train, idxs_test), batch_size=30, subject=3)

# %%
# can also use next(iter(train_loader))
for batch in train_loader:
    print(batch[0].shape, batch[1].shape, batch[2].shape)
