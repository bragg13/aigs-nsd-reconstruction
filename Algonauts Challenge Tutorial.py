# %%
# import libraries
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

# download the dataset
data_dir = './data'
subject = 3

# class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

args = argObj(data_dir, subject)

# load the dataset
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

print('LH training fMRI data shape:')
print(lh_fmri.shape)
print('(Training stimulus images × LH vertices)')

print('\nRH training fMRI data shape:')
print(rh_fmri.shape)
print('(Training stimulus images × RH vertices)')

# load images and define some params
hemispheres = ['left', 'right']
roi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
img = 0

# train image directory
train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

# train and test i
train_img_list = os.listdir(train_img_dir)
train_img_list.sort()
test_img_list = os.listdir(test_img_dir)
test_img_list.sort()

# show smth
def visualisation_brain_and_image(img=0, hemisphere='left', roi='V1v'):
    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img+1));
    plt.show()

    # Define the ROI class based on the selected ROI
    roi_class = ''
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'

    # Load the ROI brain surface maps
    challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
    roi_map_dir = os.path.join(args.data_dir, 'roi_masks',
        'mapping_'+roi_class+'.npy')
    challenge_roi_class = np.load(challenge_roi_class_dir)
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_roi))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_roi)[0]] = \
            lh_fmri[img,np.where(challenge_roi)[0]]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_roi)[0]] = \
            rh_fmri[img,np.where(challenge_roi)[0]]

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.view_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_response,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title=roi+', '+hemisphere+' hemisphere'
        )
    return view

# %% Visualise the brain surface map and the image
view = visualisation_brain_and_image(img=img, hemisphere='left', roi='V1v')
view.open_in_browser()
