# %% imports in use
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
from roi import ROI_TO_CLASS, CLASS_TO_ROI, load_roi_data
from logger import log

# %% download the dataset, this is also in nsd_data
data_dir = '../data'
subjects = [1,2,3,4,5,6,7,8]
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage')

# the class to load and store data
class argObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

SUBJECTS = dict()
for subject in subjects:
    args = argObj(data_dir, subject)
    SUBJECTS[subject] = dict()
    # load the dataset
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    SUBJECTS[subject]['lh_fmri'] = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    SUBJECTS[subject]['rh_fmri'] = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    SUBJECTS[subject]['img_dir'] = train_img_dir
    SUBJECTS[subject]['img_list'] = sorted(os.listdir(train_img_dir))

# %% img plot
def plot_img(subj: int, img: int, fig=None):
    # Load the image
    img_dir = os.path.join(SUBJECTS[subj]['img_dir'] , SUBJECTS[subj]['img_list'][img])
    train_img = Image.open(img_dir).convert('RGB')

    # Plot the image
    if fig is None:
        plt.figure()
        plt.axis('off')
        plt.imshow(train_img)
        plt.title('Training image: ' + str(img+1))
        plt.show()
    else:
        ax = fig.subplots(1, 1)
        ax.imshow(train_img)
        ax.axis('off')

# plot_img(3, 312)

# %% print data for paper figure
def print_data(subject, roi_map, challenge_roi_class, fsaverage_roi_class, start=250, end=300):
    print(f'\nfloc-bodies: {roi_map}')
    print(f'fsaverage space: {len(fsaverage_roi_class)}')
    print(f'\nlh.floc-bodies_fsaverage_space[{start}:{end}]')
    print(fsaverage_roi_class[start:end])
    print(f'challenge space: {len(challenge_roi_class)}')
    print(f'\nlh.floc-bodies_challenge_space[{start}:{end}]')
    print(challenge_roi_class[start:end])
    print(f'\nlh_fmri[0][{start}:{end}]')
    print(f'{SUBJECTS[subject]["lh_fmri"][0][start:end]}')

# %% show roi on brain surface map

# not sure where to place this correctly so it doesn't get reloaded everytime
def get_roi_data(subj, roi_class, hemi):
    roi_data = load_roi_data(subj)
    challenge_roi_class = roi_data['challenge'][hemi][roi_class] # roi indices mapped to challange space
    fsaverage_roi_class = roi_data['fsaverage'][hemi][roi_class] # roi indices mapped to fsaverage
    roi_map = roi_data['mapping'][roi_class]['id_to_roi']
    return challenge_roi_class, fsaverage_roi_class, roi_map

# map challenge space to fsaverage for roi of interest
def map_fsaverage_resp(subj, fmri, img, roi, hemisphere: str, full_class=False):
    challenge_roi_class, fsaverage_roi_class, roi_map = get_roi_data(subj, ROI_TO_CLASS[roi], hemisphere)
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
        surf_mesh=FSAVERAGE['flat' + '_' + hemi], # flat_ , pial_ , sphere_
        surf_map=fsaverage_map,
        bg_map=FSAVERAGE['sulc_' + hemi],
        threshold=1e-14,
        cmap=cmap,
        colorbar=True,
        vmax=vmax,
        vmin=vmin,
        symmetric_cmap=sym_cmap,
        title=title + ', ' + hemi + ' hemisphere',
        darkness=0.2
        )

def plot_surf(subj, fsaverage_map, hemi, title: str, cmap, style='flat', fig=None, vmax=None, vmin=None, sym_cmap=True):
    """Arg title should be ROI or ROI class"""
    if style == 'flat': view = (90, -90)
    elif (style == 'infl' or style == 'sphere') and hemi == 'lh': view = (-30, -120)
    elif (style == 'infl' or style == 'sphere') and hemi == 'rh': view = (-30, -60)
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
        title_font_size=10,
        view=view, # {“lateral”, “medial”, “dorsal”, “ventral”, “anterior”, “posterior”} or pair (0, -180.0)
        hemi=hemi,
        figure=fig,
        # output_file=f"./results/subj{subj}_roi_classes_{hemi}.png"
        )

# %% plot and view functions
def viewRoiClass(subj, roi_class, hemi, cmap):
    _, fsaverage_roi_class, _ = get_roi_data(subj, roi_class, hemi)
    return view_surf(fsaverage_roi_class, hemi, f'subj {subj}, {roi_class}', cmap, vmax=4.,vmin=1.,sym_cmap=False)

def viewRoiClassValues(subj, fmri, img, roi_class, hemi, cmap):
    fsaverage_response = map_fsaverage_resp(subj, fmri, img, CLASS_TO_ROI[roi_class][0], hemi, full_class=True)
    return view_surf(fsaverage_response, hemi, roi_class, cmap)

def viewRoiValues(subj, fmri, img, roi, hemi, cmap):
    fsaverage_response = map_fsaverage_resp(subj, fmri, img, roi, hemi)
    return view_surf(fsaverage_response, hemi, roi, cmap)

def plotRoiClass(subj, roi_class, hemi, cmap, style='infl', fig=None):
    _, fsaverage_roi_class, _ = get_roi_data(subj, roi_class, hemi)
    return plot_surf(subj, fsaverage_roi_class, hemi, f'subj {subj}', cmap, style=style, fig=fig, vmax=4.,vmin=1., sym_cmap=False)

def plotRoiClassValues(subj, fmri, img, roi_class, hemi, cmap, style, fig=None):
    fsaverage_response = map_fsaverage_resp(subj, fmri, img, CLASS_TO_ROI[roi_class][0], hemi, full_class=True)
    return plot_surf(subj, fsaverage_response, hemi, title='', cmap=cmap, style=style, fig=fig)