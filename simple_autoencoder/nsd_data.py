# %% import libraries
from logger import log
import tensorflow_datasets as tfds
import os
import numpy as np
import jax.numpy as jnp
import pandas as pd
import coco_load as cl
from jax import random
from sklearn.model_selection import train_test_split
from roi import load_roi_data



# %% track image list indices in dataframe
def images_to_nsd_df(subject=3):
    # training and test images list, sorted
    images_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_images")
    images = sorted(os.listdir(images_path))

    # make a dataframe with mapping image-nsd_index
    images_to_nsd= {}
    for i, filename in enumerate(images):
        start_i = filename.find("nsd-") + len("nds-")
        nsd_index = int(filename[start_i : start_i + 5])
        images_to_nsd[i] = [i, nsd_index]
    images_to_nsd = pd.DataFrame.from_dict(
        images_to_nsd, orient="index", columns=["listIdx", "nsdId"] # we need listIdx for the shared indices

    )
    # log(f"total images for subject {subject}: {len(images_to_nsd)}", 'DATA')
    return images_to_nsd

# %% get indices for the training, test and analysis split
def get_shared_indices(category: str = 'person', subject: int = 3):
    images_to_nsd = images_to_nsd_df(subject=subject)
    coco_loaded = cl.nsd_coco
    shared_df = cl.getSharedDf(coco_loaded).merge(images_to_nsd, on='nsdId')
    shared_category, shared_not_category = cl.splitByCategory(shared_df, category)
    # category and not indices
    # print(shared_category[:5])
    # print(shared_not_category[:5])
    category_idxs = shared_category["listIdx"].values
    not_category_idxs = shared_not_category["listIdx"].values
    return category_idxs, not_category_idxs

def get_train_test_indices(subject=3):
    """
    Get the image indices for training and testing sets for a given subject.

    Args:
        subject (int, optional): Subject ID number (1-8). Defaults to 3.

    Returns:
        tuple: Two numpy arrays containing train and test indices respectively:
            - train_idxs (np.ndarray): Indices for training set (90% of data)
            - test_idxs (np.ndarray): Indices for test set (10% of data)
    """

    # map coco categories to the pics in the dataset
    images_to_nsd = images_to_nsd_df(subject)
    coco_loaded = cl.nsd_coco
    subject_coco_df = cl.getSubjDf(coco_loaded, subject)
    subject_images = pd.merge(images_to_nsd, subject_coco_df, left_on="nsdId", right_on="nsdId", how="inner")

    train_idxs, test_idxs = train_test_split(np.arange(len(subject_images)), test_size=0.1, random_state=42)
    return train_idxs, test_idxs

# %% normalise data
# def normalise(_max, _min, data):
#    return jnp.array((data - _min) / (_max - _min))
def z_score(data):
    normalised = jnp.array((data - data.mean()) / data.std())
    return normalised

# %% get dataset splits based on indices
def get_split_masked_datasets(indices, subject, roi_class='floc-bodies', hem='all') -> tuple:
    """Get training and test fMRI datasets for a specified subject and ROI class.

    Args:
        subject (int, optional): The subject ID number (1-8). Defaults to 3.
        roi_class (str, optional): Region of interest class name. Defaults to 'floc-bodies'.
        hem (str, optional): Hemisphere selection ('all', 'lh', or 'rh'). Defaults to 'all'.

    Returns:
        tuple: Two arrays containing train and test fMRI data respectively:
            - train_fmri: Training fMRI data array
            - test_fmri: Test fMRI data array
            For hem='all', arrays contain concatenated data from both hemispheres.
            For hem='lh'/'rh', arrays contain data from specified hemisphere only.
    """
    log('creating datasets...', 'DATA')
    # get the paths to the fmri data
    fmri_base_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_fmri")
    lh_fmri_path = os.path.join(fmri_base_path, "lh_training_fmri.npy")
    rh_fmri_path = os.path.join(fmri_base_path, "rh_training_fmri.npy")

    # get the indices for training and testing
    first_idxs, sec_idxs = indices

    # load the fmri data, sliced by indexes
    # ndr: for one image, there is both the left and right hemisphere (I mean not necessarily, but yea)
    # ndr pt2: first and sec is kind of saying train and test, but it's not really that
    first_lh_fmri = jnp.load(lh_fmri_path)[first_idxs]
    first_rh_fmri = jnp.load(rh_fmri_path)[first_idxs]

    sec_lh_fmri = jnp.load(lh_fmri_path)[sec_idxs]
    sec_rh_fmri = jnp.load(rh_fmri_path)[sec_idxs]

    # get the ROI mask
    roi_data = load_roi_data(subject)
    # maske the data by ROI
    roi_lh, roi_rh = roi_data['challenge']['lh'][roi_class] > 0, roi_data['challenge']['rh'][roi_class] > 0

    first_lh_fmri = first_lh_fmri[:, roi_lh]
    first_rh_fmri = first_rh_fmri[:, roi_rh]
    sec_lh_fmri = sec_lh_fmri[:, roi_lh]
    sec_rh_fmri = sec_rh_fmri[:, roi_rh]

    if hem == 'all':
        train_all_fmri = np.concatenate([first_lh_fmri, first_rh_fmri], axis=1)
        test_all_fmri = np.concatenate([sec_lh_fmri, sec_rh_fmri], axis=1)
        return z_score(train_all_fmri), z_score(test_all_fmri)

    elif hem == 'lh':
        return z_score(first_lh_fmri), z_score(sec_lh_fmri)

    elif hem == 'rh':
        return z_score(first_rh_fmri), z_score(sec_rh_fmri)

    else:
        raise ValueError(f"Invalid hemisphere selection: {hem}. Must be 'all', 'lh', or 'rh'.")

def get_train_test_datasets(subject, roi_class='floc-bodies', hem='all') -> tuple:
    """
    Args:
        subject (int, optional): The subject ID number (1-8). Defaults to 3.
        roi_class (str, optional): Region of interest class name. Defaults to 'floc-bodies'.
        hem (str, optional): Hemisphere selection ('all', 'lh', or 'rh'). Defaults to 'all'.
    Returns:
        tuple: Two arrays containing train and test fMRI data respectively.
    """
    indices = get_train_test_indices(subject)
    return get_split_masked_datasets(indices, subject, roi_class, hem)

# aka get shared-images fmri data?
def get_analysis_datasets(category, subject, roi_class='floc-bodies', hem='all') -> tuple:
    """
    Args:
        category (str, optional): The coco category. # Defaults to 'person'.
        subject (int, optional): The subject ID number (1-8). Defaults to 3.
        roi_class (str, optional): Region of interest class name. Defaults to 'floc-bodies'.
        hem (str, optional): Hemisphere selection ('all', 'lh', or 'rh'). Defaults to 'all'.
    Returns:
        tuple: Two arrays containing fMRI data resulted from shared images of category and not-category respectively.
    """
    indices = get_shared_indices(category, subject) # category and not category indices
    return get_split_masked_datasets(indices, subject, roi_class, hem)

# %%
def get_train_test_mnist():
    # load the mnist dataset from tfds
    mnist = tfds.load("mnist", split='train')
    x_data = jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).reshape(-1, 28*28)
    x_data = x_data / 255.0
    print(f"mnist shape: {x_data.shape}")
    train, test = train_test_split(np.arange(len(x_data)), test_size=0.2, random_state=42)
    print(f"train_ds length: {len(train)}, test_ds {len(test)}")
    return x_data[train], x_data[test]

def get_train_test_cifar10():
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    # load the mnist dataset from tfds
    cifar = tfds.load("cifar10", split='train')
    x_data = jnp.array([rgb2gray(x["image"]) for x in tfds.as_numpy(cifar)]).reshape(-1, 32*32)
    x_data = x_data / 255.0
    print(f"cifar shape: {x_data.shape}")
    train, test = train_test_split(np.arange(len(x_data)), test_size=0.2, random_state=42)
    print(f"train_ds length: {len(train)}, test_ds {len(test)}")
    return x_data[train], x_data[test]

def get_batches(fmri, key, batch_size: int):
    """Create batches of fMRI data with the specified batch size.

    Args:
        fmri: Array containing fMRI data to be batched
        batch_size (int): Size of each batch

    Yields:
        ndarray: Batch of fMRI data with shape (batch_size, voxels)
    """

    num_samples = fmri.shape[0]
    permutation = random.permutation(key, num_samples // batch_size * batch_size)
    # print(f"permutatin first: {permutation[:5]}")
    return fmri[permutation]

# %% split data into lh and rh
# split_idx needs to be the len of lh fmri with only roi class, probably different for each subject
def split_hemispheres(fmri, split_idx=3633):
    return fmri[:, :split_idx], fmri[:, split_idx:]

# %% unmask data to original challenge space
# mqyabe not really unsmaking but called it that way..
def unmask_hemisphere(masked, roi_mask, challenge_fmri_shape):
    unmasked = np.zeros(challenge_fmri_shape)
    unmasked[:, roi_mask] = masked
    return unmasked

# train_all, test_all = get_train_test_datasets(hem="all")
# lh_masked, rh_masked = split_hemispheres(test_all)

# lh_unmasked = unmask_hemisphere(lh_masked, roi_lh, (819,19004))

# from surf_plot import viewRoiClassValues

# view = viewRoiClassValues(lh_unmasked, 2, 'EBA', 'lh', 'cold_hot')
# view.open_in_browser()
# %%
