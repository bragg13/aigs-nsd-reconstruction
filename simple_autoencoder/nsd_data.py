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

# %% class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+ self.subj)

def get_dir(folder: str):
    data_dir = '../data'
    subject = 3
    args = argObj(data_dir, subject)
    return os.path.join(args.data_dir, 'training_split', folder)

# rand_seed = np.random.seed(5) #@param # from colab tutorial
rand_seed = jdl.manual_seed(1234) # from jdl documentation

# %% Create the training, validation and test partitions indices
def shuffled_idxs():

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

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('Test stimulus images: ' + format(len(idxs_test)))
    return idxs_train, idxs_test

# %% pytorch dataset builder

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
    def __len__(self):
        return len(self.imgs_paths)
    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img)#.to(device)
        return img

# %% jax dataloader; returns dataloaders
# https://pypi.org/project/jax-dataloader/0.0.2/
def stim_loader(idxs, batch_s):

    def data_loader(idxs, batch_size):
        transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x24 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        dataset = ImageDataset(train_imgs_paths, idxs, transform)
        print(dataset)
        return jdl.DataLoader(
            dataset=dataset,
            backend='pytorch', # Use 'jax' backend for loading data
            batch_size=batch_size, # Batch size 
            shuffle=False, # False bec needs to match fmri ? Shuffle the dataloader every iteration or not
            drop_last=False, # Drop the last batch or not
        )

    train_imgs_paths = sorted(list(Path(get_dir('training_images')).iterdir()))
    idxs_train, idxs_test = idxs 

    train_stim_loader = data_loader(idxs_train, batch_s)
    test_stim_loader = data_loader(idxs_test, batch_s)

    return train_stim_loader, test_stim_loader

# %% load and split fmri; returns jnp arrays
def fmri_loader(idxs):
    # how to initialize this correctly, args?

    def load(fmri_dir):
        lh_fmri = jnp.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = jnp.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        print('\nLH fMRI data shape:')
        print(lh_fmri.shape, ' (Stimulus images × LH vertices)')
        print('\nRH fMRI data shape:')
        print(rh_fmri.shape, ' (Stimulus images × RH vertices)')
        
        return lh_fmri, rh_fmri

    def split(fmri_data, idxs):
        lh_fmri, rh_fmri = fmri_data
        idxs_train, idxs_test = idxs

        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_test = lh_fmri[idxs_test]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_test = rh_fmri[idxs_test]

        del lh_fmri, rh_fmri # delete to free up RAM

        return [lh_fmri_train, rh_fmri_train], [lh_fmri_test, rh_fmri_test]

    return split(load(get_dir('training_fmri')), idxs)

# %% debug
idxs = shuffled_idxs()
# fmri_train, fmri_test = fmri_loader(idxs)
stim_train, stim_test = stim_loader(idxs, 30)

# this give a TypeError: '_SingleProcessDataLoaderIter' object is not callable
# batch = next(iter(stim_train))

# %% pythorch dataloaders 
# # The DataLoaders contain the ImageDataset class
# train_imgs_dataloader = DataLoader(
#     ImageDataset(train_imgs_paths, idxs_train, transform),
#     batch_size=batch_size
# )
# test_imgs_dataloader = DataLoader(
#     ImageDataset(train_imgs_paths, idxs_test, transform),
#     batch_size=batch_size
# )