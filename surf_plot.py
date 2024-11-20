# %% imports not in use
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from pathlib import Path
from tqdm import tqdm
import matplotlib
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
# %% imports in use
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
debug = True

# %% download the dataset, this is also in nsd_data
data_dir = './data'
subject = 3

# the class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

args = argObj(data_dir, subject)

# load the dataset
fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

print(f'LH training fMRI data shape:\n{lh_fmri.shape} (Training stimulus images × LH vertices)')
print(f'\nRH training fMRI data shape:\n{rh_fmri.shape} (Training stimulus images × RH vertices)')

# %% load image lists and define some params
hemispheres = ['left', 'right']
roi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
img = 0

# train image directory
train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

# train and test i
train_img_list = os.listdir(train_img_dir).sort()
test_img_list = os.listdir(test_img_dir).sort()

# %% img plot and roi class def
def plotImg(img: int):
    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img+1))
    plt.show()

def get_roi_class(roi: str):
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
    return roi_class

# %% convert numpy array to nii image object
def conv_to_nii(fmri_arr):
    example_file = os.path.join('data/test/T2_to_MNI.nii.gz')
    exImg = nib.load(example_file) # shape (120, 120, 84, 188)
    nii = image.new_img_like(exImg, fmri_arr)
    print(nii.shape)
    return nii

# %% show roi on brain surface map
def visualisation_brain_and_image(img=0, hemisphere='left', roi='EBA', full_class=False, cmap='cold_hot'):
    # if not debug: plotImg(img) # some bug, not super relevant
    roi_class = get_roi_class(roi)

    # Load the ROI brain surface maps
    challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks', hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks', hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
    roi_map_dir = os.path.join(args.data_dir, 'roi_masks', 'mapping_'+roi_class+'.npy')
    
    challenge_roi_class = np.load(challenge_roi_class_dir) # mapped roi indices to brain voxels
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # try converting np array to nii image for visualisation
    # niiTest = conv_to_nii(fsaverage_roi_class)
    
    if debug:
        start = 250
        end = 300
        print(f'\nfloc-bodies: {roi_map}')
        # print(f'fsaverage space: {len(fsaverage_roi_class)}')
        # print(f'\nlh.floc-bodies_fsaverage_space[{start}:{end}]')
        # print(fsaverage_roi_class[start:end])
        print(f'challenge space: {len(challenge_roi_class)}')
        print(f'\nlh.floc-bodies_challenge_space[{start}:{end}]')
        print(challenge_roi_class[start:end])
        print(f'\nlh_fmri[0][{start}:{end}]')
        print(f'{lh_fmri[0][start:end]}')

    # map only one roi or the roi class to fsaverage
    def map_fsaverage_resp(hemisphere:str, all_rois=False):
        if all_rois: fsvg_roi, ch_roi = fsaverage_roi_class, challenge_roi_class
        else:
            # Select the vertices corresponding to the ROI of interest
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)] # the id of the roi
            challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int) # set all other roi ids to 0 (False)
            fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
            fsvg_roi, ch_roi = fsaverage_roi, challenge_roi
        
        fsaverage_response = np.zeros(len(fsvg_roi))
        if hemisphere == 'left':
            fsaverage_response[np.where(fsvg_roi)[0]] = lh_fmri[img,np.where(ch_roi)[0]]
        elif hemisphere == 'right':
            fsaverage_response[np.where(fsvg_roi)[0]] = rh_fmri[img,np.where(ch_roi)[0]]
        return fsaverage_response
    
    if full_class: 
        fsaverage_response = fsaverage_roi_class
        vmin, vmax, symm_cmap = 1.,3., False
        title = roi_class
    else: 
        fsaverage_response = map_fsaverage_resp(hemisphere)
        vmin, vmax, symm_cmap = None, None, True
        title = roi

    # Map the fMRI data onto the brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

    if not debug:
        plotting.plot_anat()
        view = plotting.view_surf(
            surf_mesh=fsaverage['infl_'+hemisphere], # flat_ , pial_ , sphere_
            surf_map=fsaverage_response,
            bg_map=fsaverage['sulc_'+hemisphere],
            threshold=1e-14,
            cmap=cmap,
            colorbar=True,
            title=title +', '+hemisphere+' hemisphere',
            vmax=vmax,
            vmin=vmin,
            symmetric_cmap=symm_cmap
            )
        return view

# Visualise the brain surface map and the image
if debug: visualisation_brain_and_image(img=img, hemisphere='left', roi='EBA', cmap='tab10')
else:
    view_class = visualisation_brain_and_image(img=img, hemisphere='left', roi='EBA', full_class=True, cmap='brg')
    view_class.open_in_browser()
    view1 = visualisation_brain_and_image(img=img, hemisphere='left', roi='EBA', cmap='blue_transparent_full_alpha_range')
    view1.open_in_browser()
    # view2 = visualisation_brain_and_image(img=img, hemisphere='left', roi='FBA-1', cmap='red_transparent_full_alpha_range')
    # view2.open_in_browser()
    # view3 = visualisation_brain_and_image(img=img, hemisphere='left', roi='FBA-2', cmap='green_transparent_full_alpha_range')
    # view3.open_in_browser()
    # view4 = visualisation_brain_and_image(img=img, hemisphere='left', roi='mTL-bodies', cmap='Wistia')
    # view4.open_in_browser()
