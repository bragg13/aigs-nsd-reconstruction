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

# %% class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+ self.subj)

def get_dir(folder: str, subject=3):
    data_dir = '../data'
    args = argObj(data_dir, subject)
    return os.path.join(args.data_dir, 'training_split', folder)

subject = 3
rand_seed = jdl.manual_seed(1234) # from jdl documentation

# TODO: finish the idexes splitting (later)
# %% store img idx and corresponding nsd id in dataframe
def make_imgs_df():
    def make_stim_list(stim_dir):
        # Create lists will all training and test image file names, sorted
        stim_list = os.listdir(stim_dir)
        stim_list.sort()
        return stim_list
    stim_list = make_stim_list(get_dir('training_images'))

    # make dictionary where the images' list index is mapped to the nsd id
    stim_nsd_idxs = {}
    for i, filename in enumerate(stim_list):
        start_i = filename.find('nsd-') + len('nds-')
        nsd_index = int(filename[start_i:start_i + 5])
        stim_nsd_idxs[i] = [i, nsd_index]
    
    # convert dictionary to dataframe
    stim_nsd_idxs_df = pd.DataFrame.from_dict(stim_nsd_idxs, orient='index', columns=['listIdx','nsdId'])
    print(f'total algonauts for subj{subject}: {len(stim_nsd_idxs_df)}')
    return stim_nsd_idxs_df

# %% split indices into training (90% of subject specfic), test (10% of subject specific), person ()
def split_idxs(category='person'):
    indices = {}
    img_df = make_imgs_df()

    def merge(df):
        return pd.merge(df, img_df, left_on='nsdId', right_on='nsdId', how='inner')
    
    nsd_coco = cl.read_and_preprocess()
    subj_df = merge(cl.getSubjDf(nsd_coco, subject))
    shared_df = merge(cl.getSharedDf(nsd_coco))
    shared_pers, shared_not_pers = cl.splitByCategory(shared_df, category)

    # training and test indices (90%/10%) 
    subj_idxs = subj_df['listIdx'].values
    num_train = int(np.round(len(subj_idxs) / 100 * 90))
    # np.random.shuffle(subj_idxs) # not sure if this is necessary
    indices['train'] = subj_idxs[:num_train]
    indices['test'] = subj_idxs[num_train:]

    # category and not indices
    indices[category] = shared_pers['listIdx'].values
    indices[f'not_{category}'] = shared_not_pers['listIdx'].values

    print(f'\nalgonauts:')
    print(f'columns: {shared_df.columns}')
    print(f'shared: {len(indices[category])} + {len(indices[f"not_{category}"])} = {len(shared_df)} images')
    print(f'subj{subject}: {len(indices["train"])} + {len(indices["test"])} = {len(subj_df)} images')
    print(f'idx of subj{subject} train split: {indices["train"][0:10]} etc')
    print(f'idx of subj{subject} test split: {indices["test"][0:10]} etc')
    print(f'idx of shared "{category}": {indices[category][0:10]} etc')
    print(f'idx of shared not "{category}": {indices[f"not_{category}"][0:10]} etc')

    return indices

indices = split_idxs()

# %% old
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
# def custom_transforms(image):
#     # Resize to 224x224
#     image = image.resize((224, 224))
#     # image = image.resize((224, 224))

#     # convert and normalize
#     image_array = np.array(image) / 255.0
#     image_array = (image_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize

#     # Rearrange dimensions to match JAX's convention (HWC)
#     image_array = np.transpose(image_array, (2, 0, 1))  # Change to (C, H, W) if needed for your model
#     image_array = image_array.reshape((224, 224, 3))
#     return image_array

# TODO: improve dataset to just contain fmri - remove images
class ImageAndFmriDataset(jdl.Dataset):
    """ Dataset class for loading images and fMRI data """
    def __init__(self, fmri_paths, fmri_roi, idxs):
        # use
        self.idxs = idxs
        self.lh_fmri = jnp.load(fmri_paths[0])[idxs, 0:1000] # 0:10 fmri. if indexes, gnna be different
        self.rh_fmri = jnp.load(fmri_paths[1])[idxs, 0:1000]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # data = np.concat([self.lh_fmri[idx], self.rh_fmri[idx]])
        # data = np.vstack((self.lh_fmri[idx], self.rh_fmri[idx]))
        # print(data.shape)
        return np.concatenate([self.lh_fmri[idx], self.lh_fmri[idx]], axis=1) # self.lh_fmri[idx]

# %% main
# TODO: implement ROI masking - remove images
def create_loaders(all_idxs, batch_size, roi, subject=3):
    # directories - we use shared images for testing, so everything is in the same directory
    imgs_dir = get_dir('training_images', subject)
    imgs_paths = sorted(list(Path(imgs_dir).iterdir())) # could pass directly this

    fmri_dir = get_dir('training_fmri', subject)
    lh_fmri_path = os.path.join(fmri_dir, 'lh_training_fmri.npy')
    rh_fmri_path = os.path.join(fmri_dir, 'rh_training_fmri.npy')
    fmri_paths = [lh_fmri_path, rh_fmri_path]

    # imgs = sorted(list(Path(self.imgs_dir).iterdir()))
    # imgs = []
    # for i in range(len(imgs_paths)):
    #     img = custom_transforms(Image.open(imgs_paths[i]).convert('RGB'))
    #     if i % 200 == 0:
    #         print(i)
    #     imgs.append(img)
    # imgs = jnp.array(imgs)

    # indexes - what actually changes
    idxs_train, idxs_test = all_idxs

    train_dataset = ImageAndFmriDataset(fmri_paths, roi, idxs_train)
    test_dataset = ImageAndFmriDataset(fmri_paths, roi, idxs_test)

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
# idxs_train, idxs_test = shuffle_idxs()
# train_loader, test_loader = create_loaders((idxs_train, idxs_test), batch_size=30, roi=None, subject=3)

# %%
# batch = next(iter(train_loader))
# batch.shape # (4,1000) 4 is the number of fmri pictures (will be 9000) and 2000 is the number of voxels (will depend on the ROI)
# %%
# print(len(train_loader))