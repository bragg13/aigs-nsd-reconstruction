# import libraries
from logger import log
import os
import numpy as np
from pathlib import Path
import jax.numpy as jnp
from tqdm.gui import tqdm
import pandas as pd
import coco_load as cl
import nsd_data
import matplotlib.pyplot as plt

# from visualisations import plot_data_distribution
from sklearn.model_selection import train_test_split
from roi import load_roi_data

# %% class to load and store data

# %% split indices into training (90% of subject specfic), test (10% of subject specific), person ()
# def split_idxs(category="person") -> dict[str, list[int]]:
#     indices = {}
#     img_df = make_imgs_df()
#     print(img_df.head())

#     def merge(df):
#         return pd.merge(df, img_df, left_on="nsdId", right_on="nsdId", how="inner")

#     coco_loaded = cl.nsd_coco
#     subj_df = merge(cl.getSubjDf(coco_loaded, subject))
#     shared_df = merge(cl.getSharedDf(coco_loaded))
#     shared_pers, shared_not_pers = cl.splitByCategory(shared_df, category)

#     # training and test indices (90%/10%)
#     subj_idxs = subj_df["listIdx"].values
#     num_train = int(np.round(len(subj_idxs) / 100 * 90))
#     # np.random.shuffle(subj_idxs) # not sure if this is necessary
#     indices["subject_train"] = subj_idxs[:num_train]
#     indices["subject_test"] = subj_idxs[num_train:]

#     # category and not indices
#     indices[f"shared_{category}"] = shared_pers["listIdx"].values
#     indices[f"shared_not_{category}"] = shared_not_pers["listIdx"].values

#     if debug:
#         print()
#         print("algonauts:")
#         print(f"columns: {shared_df.columns}")
#         print(
#             f'shared: {len(indices[f"shared_{category}"])} + {len(indices[f"shared_not_{category}"])} = {len(shared_df)} images'
#         )
#         print(
#             f'subj{subject}: {len(indices["subject_train"])} + {len(indices["subject_test"])} = {len(subj_df)} images'
#         )
#         print(f'idx of subj{subject} train split: {indices["subject_train"][0:10]} etc')
#         print(f'idx of subj{subject} test split: {indices["subject_test"][0:10]} etc')
#         print(f'idx of shared "{category}": {indices[f"shared_{category}"][0:10]} etc')
#         print(f'idx of shared not "{category}": {indices[f"shared_not_{category}"][0:10]} etc')

#     return indices



# class FmriDataset(jdl.Dataset):
#     """Dataset class for loading images and fMRI data"""

#     def __init__(self, fmri_paths, idxs, roi, hem):
#         self.idxs = idxs # how am i pssing hte idxs?
#         self.hem = hem
#         self.fmri = None
#         roi_lh, roi_rh = get_roi(roi)

#         if self.hem == 'all' or self.hem == 'rh':
#             rh_fmri = jnp.load(fmri_paths[1])[idxs]
#             rh_max = rh_fmri.max()
#             rh_min = rh_fmri.min()
#             print(f"max rh fmri value: {rh_max}")
#             print(f"min rh fmri value: {rh_min}")
#             self.rh_fmri = rh_fmri[:, roi_rh]

#         if self.hem == 'all' or self.hem == 'lh':
#             lh_fmri = jnp.load(fmri_paths[0])[idxs]
#             lh_max = lh_fmri.max()
#             lh_min = lh_fmri.min()
#             print(f"max lh fmri value: {lh_max}")
#             print(f"min lh fmri value: {lh_min}")
#         roi_lh, roi_rh = get_roi(roi)
#             self.lh_fmri = lh_fmri[:, roi_lh]


#     def __len__(self):
#         return len(self.idxs)

#     def get_fmri_shape(self):
#         if self.hem == 'all':
#             return self.lh_fmri.shape, self.rh_fmri.shape
#         if self.hem == 'rh':
#             return self.rh_fmri.shape
#         if self.hem == 'lh':
#             return self.lh_fmri.shape

#     def get_fmri_voxels(self):
#         if self.hem == 'all':
#             return self.lh_fmri.shape[1] + self.rh_fmri.shape[1]
#         if self.hem == 'rh':
#             return self.rh_fmri.shape[1]
#         if self.hem == 'lh':
#             return self.lh_fmri.shape[1]

#     def __getitem__(self, idx):
#         if self.hem == 'all':
#             return np.concatenate([self.lh_fmri[idx], self.lh_fmri[idx]], axis=1)
#         if self.hem == 'rh':
#             return self.rh_fmri[idx]
#         if self.hem == 'lh':
#             return self.lh_fmri[idx]



# %% main
# def create_loaders(all_idxs, batch_size, roi, hem, subject=3):
#     # indexes - what actually changes
#     idxs_train, idxs_test = all_idxs

#     log('Initialising training dataset', 'DATASET')
#     train_dataset = FmriDataset(fmri_paths, idxs_train, roi, hem, train=True)
#     print(f"length: {len(train_dataset)}, fmri voxels: {train_dataset.get_fmri_voxels()}")

#     log('Initialising test dataset', 'DATASET')
#     test_dataset = FmriDataset(fmri_paths,  idxs_test, roi, hem, train=False)
#     print(f"length: {len(test_dataset)}, fmri voxels: {test_dataset.get_fmri_voxels()}")

#     train_size = len(train_dataset)
#     test_size = len(test_dataset)

#     train_loader = jdl.DataLoader(
#         dataset=train_dataset,
#         backend="jax",
#         batch_size=batch_size,
#         shuffle=True,  # now we can shuffle cause we have tuples
#         drop_last=False,
#     )
#     test_loader = jdl.DataLoader(
#         dataset=test_dataset,
#         backend="jax",
#         batch_size=batch_size,
#         shuffle=True,  # now we can shuffle cause we have tuples
#         drop_last=False,
#     )
#     voxels = train_dataset.get_fmri_voxels() #train_dataset.get_fmri_shape()[1]
#     return train_loader, test_loader, train_size, test_size, voxels
#     # jnp.fft.fft2(matirx)



def get_shared_indices():
    # shared_df = merge(cl.getSharedDf(coco_loaded))
    # shared_pers, shared_not_pers = cl.splitByCategory(shared_df, category)
    # # category and not indices
    # indices[f"shared_{category}"] = shared_pers["listIdx"].values
    # indices[f"shared_not_{category}"] = shared_not_pers["listIdx"].values
    pass

def get_train_test_indexes(subject=3):
    """
    Get the image indices for training and testing sets for a given subject.

    Args:
        subject (int, optional): Subject ID number (1-8). Defaults to 3.

    Returns:
        tuple: Two numpy arrays containing train and test indices respectively:
            - train_idxs (np.ndarray): Indices for training set (90% of data)
            - test_idxs (np.ndarray): Indices for test set (10% of data)
    """
    # training and test images list, sorted
    images_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_images")
    images = sorted(os.listdir(images_path))

    # make a dataframe with mapping image-nsd_index
    images_to_nsd= {}
    for i, filename in enumerate(images):
        start_i = filename.find("nsd-") + len("nds-")
        nsd_index = int(filename[start_i : start_i + 5])
        images_to_nsd[i] = [nsd_index]
    images_to_nsd = pd.DataFrame.from_dict(
        images_to_nsd, orient="index", columns=["nsdId"]
    )
    print(f"total images for subject {subject}: {len(images_to_nsd)}")

    # map coco categories to the pics in the dataset
    coco_loaded = cl.nsd_coco
    subject_coco_df = cl.getSubjDf(coco_loaded, subject)
    subject_images = pd.merge(images_to_nsd, subject_coco_df, left_on="nsdId", right_on="nsdId", how="inner")

    train_idxs, test_idxs = train_test_split(np.arange(len(subject_images)), test_size=0.1, random_state=42)
    return train_idxs, test_idxs


def get_train_test_datasets(subject=3, roi_class='floc-bodies', hem='all') -> tuple:
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
    print('getting datasets...')
    # get the paths to the fmri data
    fmri_base_path = os.path.join("../data", "subj0"+str(subject), "training_split", "training_fmri")
    lh_fmri_path = os.path.join(fmri_base_path, "lh_training_fmri.npy")
    rh_fmri_path = os.path.join(fmri_base_path, "rh_training_fmri.npy")

    # get the indices for training and testing
    train_idxs, test_idxs = get_train_test_indexes(subject)

    # get the ROI mask
    roi_data = load_roi_data(subject=3)

    # load the fmri data, sliced by indexes
    # ndr: for one image, there is both the left and right hemisphere (I mean not necessarily, but yea)
    train_lh_fmri = jnp.load(lh_fmri_path)[train_idxs]
    train_rh_fmri = jnp.load(rh_fmri_path)[train_idxs]

    test_lh_fmri = jnp.load(lh_fmri_path)[test_idxs]
    test_rh_fmri = jnp.load(rh_fmri_path)[test_idxs]

    # maske the data by ROI
    roi_lh, roi_rh = roi_data['challenge']['lh'][roi_class] > 0, roi_data['challenge']['rh'][roi_class] > 0

    train_lh_fmri = train_lh_fmri[:, roi_lh]
    train_rh_fmri = train_rh_fmri[:, roi_rh]
    test_lh_fmri = test_lh_fmri[:, roi_lh]
    test_rh_fmri = test_rh_fmri[:, roi_rh]

    # print(f"train_lh_fmri min: {train_lh_fmri.min()}, max: {train_lh_fmri.max()}")
    # print(f"train_rh_fmri min: {train_rh_fmri.min()}, max: {train_rh_fmri.max()}")
    # print(f"train_lh_fmri shape: {train_lh_fmri.shape}")
    # print(f"train_rh_fmri shape: {train_rh_fmri.shape}")

    # print(f"test_lh_fmri min: {test_lh_fmri.min()}, max: {test_lh_fmri.max()}")
    # print(f"test_rh_fmri min: {test_rh_fmri.min()}, max: {test_rh_fmri.max()}")
    # print(f"test_lh_fmri shape: {test_lh_fmri.shape}")
    # print(f"test_rh_fmri shape: {test_rh_fmri.shape}")

    if hem == 'all':
        train_all_fmri = np.concatenate([train_lh_fmri, train_rh_fmri], axis=1)
        test_all_fmri = np.concatenate([test_lh_fmri, test_rh_fmri], axis=1)
        return train_all_fmri, test_all_fmri
    elif hem == 'lh':
        return train_lh_fmri, test_lh_fmri
    elif hem == 'rh':
        return train_rh_fmri, test_rh_fmri
    else:
        raise ValueError(f"Invalid hemisphere selection: {hem}. Must be 'all', 'lh', or 'rh'.")

def get_batches(fmri, batch_size: int):
    """Create batches of fMRI data with the specified batch size.

    Args:
        fmri: Array containing fMRI data to be batched
        batch_size (int): Size of each batch

    Yields:
        ndarray: Batch of fMRI data with shape (batch_size, voxels)
    """

    num_samples = fmri.shape[0]
    while True:
        permutation = np.random.permutation(num_samples // batch_size * batch_size)
        for i in range(0, len(permutation), batch_size):
            batch_perm = permutation[i:i + batch_size]
            batch = fmri[batch_perm]
            # batch_volume = fmri[batch_perm] ... TODO
            yield batch
