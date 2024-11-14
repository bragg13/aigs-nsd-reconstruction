# %%
# import libraries
import os
import numpy as np
from pathlib import Path
from PIL import Image
import jax_dataloader as jdl
import jax.numpy as jnp
from tqdm.gui import tqdm
import pandas as pd
import coco_load as cl
import nsd_data
debug = False



# %% class to load and store data
class argObj:
    def __init__(self, data_dir, subj):
        self.subj = format(subj, "02")
        self.data_dir = os.path.join(data_dir, "subj" + self.subj)


def get_dir_training(folder: str, subject=3):
    data_dir = "../data"
    args = argObj(data_dir, subject)
    return os.path.join(args.data_dir, "training_split", folder)

def get_dir_roi(subject=3):
    data_dir = "../data"
    args = argObj(data_dir, subject)
    return os.path.join(args.data_dir, "roi_masks")


subject = 3
rand_seed = jdl.manual_seed(1234)  # from jdl documentation


# TODO: finish the idexes splitting (later)
# %% store img idx and corresponding nsd id in dataframe
def make_imgs_df():
    def make_stim_list(stim_dir):
        # Create lists will all training and test image file names, sorted
        stim_list = os.listdir(stim_dir)
        stim_list.sort()
        return stim_list

    stim_list = make_stim_list(get_dir_training("training_images"))

    # make dictionary where the images' list index is mapped to the nsd id
    stim_nsd_idxs = {}
    for i, filename in enumerate(stim_list):
        start_i = filename.find("nsd-") + len("nds-")
        nsd_index = int(filename[start_i : start_i + 5])
        stim_nsd_idxs[i] = [i, nsd_index]

    # convert dictionary to dataframe
    stim_nsd_idxs_df = pd.DataFrame.from_dict(
        stim_nsd_idxs, orient="index", columns=["listIdx", "nsdId"]
    )
    print(f"total: {len(stim_nsd_idxs_df)}")
    return stim_nsd_idxs_df


# %% split indices into training (90% of subject specfic), test (10% of subject specific), person ()
def split_idxs(category="person") -> dict[str, list[int]]:
    indices = {}
    img_df = make_imgs_df()
    print(img_df.head())

    def merge(df):
        return pd.merge(df, img_df, left_on="nsdId", right_on="nsdId", how="inner")

    coco_loaded = cl.nsd_coco
    subj_df = merge(cl.getSubjDf(coco_loaded, subject))
    shared_df = merge(cl.getSharedDf(coco_loaded))
    shared_pers, shared_not_pers = cl.splitByCategory(shared_df, category)

    # training and test indices (90%/10%)
    subj_idxs = subj_df["listIdx"].values
    num_train = int(np.round(len(subj_idxs) / 100 * 90))
    # np.random.shuffle(subj_idxs) # not sure if this is necessary
    indices["subject_train"] = subj_idxs[:num_train]
    indices["subject_test"] = subj_idxs[num_train:]

    # category and not indices
    indices[f"shared_{category}"] = shared_pers["listIdx"].values
    indices[f"shared_not_{category}"] = shared_not_pers["listIdx"].values

    if debug:
        print()
        print("algonauts:")
        print(f"columns: {shared_df.columns}")
        print(
            f'shared: {len(indices[f"shared_{category}"])} + {len(indices[f"shared_not_{category}"])} = {len(shared_df)} images'
        )
        print(
            f'subj{subject}: {len(indices["subject_train"])} + {len(indices["subject_test"])} = {len(subj_df)} images'
        )
        print(f'idx of subj{subject} train split: {indices["subject_train"][0:10]} etc')
        print(f'idx of subj{subject} test split: {indices["subject_test"][0:10]} etc')
        print(f'idx of shared "{category}": {indices[f"shared_{category}"][0:10]} etc')
        print(f'idx of shared not "{category}": {indices[f"shared_not_{category}"][0:10]} etc')

    return indices



# TODO: improve dataset to just contain fmri - remove images
class FmriDataset(jdl.Dataset):
    """Dataset class for loading images and fMRI data"""

    def __init__(self, fmri_paths, idxs):
        # use
        self.idxs = idxs
        roi_lh, roi_rh = get_roi()
        lh_fmri = jnp.load(fmri_paths[0])[idxs]
        rh_fmri = jnp.load(fmri_paths[1])[idxs]
        self.lh_fmri = lh_fmri[:, roi_lh]
        self.rh_fmri = rh_fmri[:, roi_rh]
        print('min max')
        print(lh_fmri.max())
        print(lh_fmri.min())

    def __len__(self):
        return len(self.idxs)

    def get_fmri_shape(self):
        return self.lh_fmri.shape, self.rh_fmri.shape

    def __getitem__(self, idx):
        # return np.concatenate([self.lh_fmri[idx], self.lh_fmri[idx]], axis=1)
        return self.lh_fmri[idx]

# %% load ROI
def get_roi():
    roi_class = 'floc-bodies'
    challenge_roi_class_dir_lh = os.path.join(get_dir_roi(),  'lh.'+roi_class+'_challenge_space.npy')
    challenge_roi_class_dir_rh = os.path.join(get_dir_roi(),  'rh.'+roi_class+'_challenge_space.npy')
    challenge_roi_class_lh = np.load(challenge_roi_class_dir_lh)
    challenge_roi_class_rh = np.load(challenge_roi_class_dir_rh)

    # Create a boolean mask for the floc-bodies ROI
    floc_bodies_mask_lh = challenge_roi_class_lh > 0
    floc_bodies_mask_rh = challenge_roi_class_rh > 0

    return floc_bodies_mask_lh, floc_bodies_mask_rh

# %% main
# TODO: implement ROI masking - remove images
def create_loaders(all_idxs, batch_size, roi, subject=3):
    # directories - we use shared images for testing, so everything is in the same directory
    imgs_dir = get_dir_training("training_images", subject)
    imgs_paths = sorted(list(Path(imgs_dir).iterdir()))  # could pass directly this

    fmri_dir = get_dir_training("training_fmri", subject)
    lh_fmri_path = os.path.join(fmri_dir, "lh_training_fmri.npy")
    rh_fmri_path = os.path.join(fmri_dir, "rh_training_fmri.npy")
    fmri_paths = [lh_fmri_path, rh_fmri_path]

    # indexes - what actually changes
    idxs_train, idxs_test = all_idxs

    train_dataset = FmriDataset(fmri_paths, idxs_train)
    print(f"train len: {len(train_dataset)}, fmri shape: {train_dataset.get_fmri_shape()}")
    test_dataset = FmriDataset(fmri_paths,  idxs_test)
    print(f"test len: {len(test_dataset)}, fmri shape: {test_dataset.get_fmri_shape()}")

    train_loader = jdl.DataLoader(
        dataset=train_dataset,
        backend="jax",
        batch_size=batch_size,
        shuffle=True,  # now we can shuffle cause we have tuples
        drop_last=False,
    )
    test_loader = jdl.DataLoader(
        dataset=test_dataset,
        backend="jax",
        batch_size=batch_size,
        shuffle=True,  # now we can shuffle cause we have tuples
        drop_last=False,
    )

    return train_loader, test_loader


# %% main
# idxs_train, idxs_test = shuffle_idxs()
# train_loader, test_loader = create_loaders((idxs_train, idxs_test), batch_size=30, roi=None, subject=3)

# %%
# batch = next(iter(train_loader))
# batch.shape # (4,1000) 4 is the number of fmri pictures (will be 9000) and 2000 is the number of voxels (will depend on the ROI)
# %%
# print(len(train_loader))
