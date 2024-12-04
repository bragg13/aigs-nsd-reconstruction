# %% imports in use
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
from roi import ROI_TO_CLASS, load_roi_data
from logger import log

# %% download the dataset, this is also in nsd_data
data_dir = '../data'
subject = 3
img = 0
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage')
print(FSAVERAGE.keys())
# the class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

args = argObj(data_dir, subject)

# # load the dataset
# fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
# lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
# rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

# print(f'LH training fMRI data shape:\n{lh_fmri.shape} (Training stimulus images × LH vertices)')
# print(f'\nRH training fMRI data shape:\n{rh_fmri.shape} (Training stimulus images × RH vertices)')

train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
train_img_list = os.listdir(train_img_dir).sort()

# %% img plot
def plot_img(img: int):
    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    plt.figure()
    plt.axis('off')
    plt.imshow(train_img)
    plt.title('Training image: ' + str(img+1))
    plt.show()

# %% print data for paper figure
def print_data(roi_map, challenge_roi_class, fsaverage_roi_class, start=250, end=300):
    print(f'\nfloc-bodies: {roi_map}')
    print(f'fsaverage space: {len(fsaverage_roi_class)}')
    print(f'\nlh.floc-bodies_fsaverage_space[{start}:{end}]')
    print(fsaverage_roi_class[start:end])
    print(f'challenge space: {len(challenge_roi_class)}')
    print(f'\nlh.floc-bodies_challenge_space[{start}:{end}]')
    print(challenge_roi_class[start:end])
    print(f'\nlh_fmri[0][{start}:{end}]')
    print(f'{lh_fmri[0][start:end]}')

# %% show roi on brain surface map

# not sure where to place this correctly so it doesn't get reloaded everytime
roi_data = load_roi_data(subject)

def get_roi_data(roi_class, hemi):
    challenge_roi_class = roi_data['challenge'][hemi][roi_class] # roi indices mapped to challange space
    fsaverage_roi_class = roi_data['fsaverage'][hemi][roi_class] # roi indices mapped to fsaverage
    roi_map = roi_data['mapping'][roi_class]['id_to_roi']
    return challenge_roi_class, fsaverage_roi_class, roi_map

# map challenge space to fsaverage for roi of interest
def map_fsaverage_resp(fmri, img, roi, hemisphere: str, full_class=False):
    challenge_roi_class, fsaverage_roi_class, roi_map = get_roi_data(ROI_TO_CLASS[roi], hemisphere)
    
    if full_class: 
        fsvg_roi, ch_roi = fsaverage_roi_class, challenge_roi_class
    else:
        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)] # the id of the roi
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int) # set all other roi ids to 0 (False)
        fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
        fsvg_roi, ch_roi = fsaverage_roi, challenge_roi
    
    fsaverage_response = np.zeros(len(fsvg_roi))
    fsaverage_response[np.where(fsvg_roi)[0]] = fmri[img, np.where(ch_roi)[0]]
    return fsaverage_response

def view_surf(fsaverage_map, hemi, title: str, cmap, vmax=None, vmin=None, sym_cmap=True):
    """
    Arg title should be ROI or ROI class
    Args:
        title (str): 
    """
    if hemi == 'lh': hemi = 'left'
    elif hemi == 'rh': hemi = 'right'
    return plotting.view_surf(
        surf_mesh=FSAVERAGE['infl' + '_' + hemi], # flat_ , pial_ , sphere_
        surf_map=fsaverage_map,
        bg_map=FSAVERAGE['sulc_' + hemi],
        threshold=1e-14,
        cmap=cmap,
        colorbar=True,
        vmax=vmax,
        vmin=vmin,
        symmetric_cmap=sym_cmap,
        title=title + ', ' + hemi + ' hemisphere'
        )

def plot_surf(fsaverage_map, hemi, title: str, cmap, style='flat', fig=None, vmax=None, vmin=None, sym_cmap=True):
    """Arg title should be ROI or ROI class"""
    if style == 'flat': view = (90, -90)
    elif style == 'infl' and hemi == 'lh': view = (-30, -120)
    elif style == 'infl' and hemi == 'rh': view = (-30, -60)
    if hemi == 'lh': hemi = 'left'
    elif hemi == 'rh': hemi = 'right'
    return plotting.plot_surf(
        surf_mesh=FSAVERAGE[style + '_' + hemi], # infl_, flat_ , pial_ , sphere_
        surf_map=fsaverage_map,
        bg_map=FSAVERAGE['area_' + hemi], # sulc_ , curv_
        threshold=1e-14,
        cmap=cmap,
        colorbar=False,
        darkness=0.2,
        vmax=vmax,
        vmin=vmin,
        symmetric_cmap=sym_cmap,
        # title= title + ', ' + hemi + ' hemisphere',
        title_font_size= 10,
        view=view, # {“lateral”, “medial”, “dorsal”, “ventral”, “anterior”, “posterior”} or pair (0, -180.0)
        hemi=hemi,
        figure=fig
        # output_file=f'./results/epoch_{epoch}.png'
        )

# %% plot and view functions
def viewRoiClass(roi_class, hemi, cmap):
    _, fsaverage_roi_class, _ = get_roi_data(roi_class, hemi)
    return view_surf(fsaverage_roi_class, hemi, roi_class, cmap, vmax=3.,vmin=1.,sym_cmap=False)

def viewRoiClassValues(fmri, img, roi, hemi, cmap):
    fsaverage_response = map_fsaverage_resp(fmri, img, roi, hemi, full_class=True)
    return view_surf(fsaverage_response, hemi, ROI_TO_CLASS[roi], cmap)

def viewRoiValues(fmri, img, roi, hemi, cmap):
    fsaverage_response = map_fsaverage_resp(fmri, img, roi, hemi)
    return view_surf(fsaverage_response, hemi, roi, cmap)

def plotRoiClassValues(fmri, img, roi, hemi, cmap, style, fig=None):
    fsaverage_response = map_fsaverage_resp(fmri, img, roi, hemi, full_class=True)
    return plot_surf(fsaverage_response, hemi, title='', cmap=cmap, style=style, fig=fig)
# %% test view in browser 
# first_test_idx = 2126 # check nsd_data
# third_test_idx = 4894 # check nsd_data
# roiclassvalues = viewRoiClassValues(lh_fmri, third_test_idx, 'EBA', hemi='lh', cmap='cold_hot')
# roiclassvalues.open_in_browser()
# roiclass = viewRoiClass('floc-bodies', hemi='lh', cmap='brg')
# roiclass.open_in_browser()
# roivalues = viewRoiValues(lh_fmri, img, 'EBA', 'lh', cmap='blue_transparent_full_alpha_range')
# roivalues.open_in_browser()

# plotRoiClassValues(rh_fmri, img, 'EBA', hemi='rh', cmap='cold_hot', style='infl')