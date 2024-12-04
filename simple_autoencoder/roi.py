#%%
import os
import numpy as np

# %% Dict mapping ROI class names to lists of ROI names
CLASS_TO_ROI = {"prf-visualrois":  ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
                    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
                    "floc-faces":  ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
                    "floc-places": ["OPA", "PPA", "RSC"],
                    "floc-words":  ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
                    "streams":     ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]}

# Dict mapping ROI names to their class name
ROI_TO_CLASS = {roi: roi_class for roi_class, rois in CLASS_TO_ROI.items() for roi in rois}

# List of all ROI names
ROIS = [roi for roi_class in CLASS_TO_ROI.values() for roi in roi_class]

def load_roi_data(subject=3):
    """
    Loads ROI data structures for specified subject
        Args:
            subject (int): Subject number (default: 3)
        Returns:
            dict: Mapping, challenge and fsaverage space ROI data
    """
    roi_dir = os.path.join('..', 'data', "subj0"+str(subject), "roi_masks")

    data = {'mapping' : {},
            'challenge' : {'lh' : {}, 'rh' : {}},
            'fsaverage' : {'lh' : {}, 'rh' : {}}}

    for roi_class in CLASS_TO_ROI.keys():
        data['mapping'][roi_class] = {'id_to_roi' : {}, 'roi_to_id' : {}}
        data['mapping'][roi_class]['id_to_roi'] = np.load(os.path.join(roi_dir, f'mapping_{roi_class}.npy'), allow_pickle=True).item()
        # do we need the below?
        data['mapping'][roi_class]['roi_to_id'] = {v: k for k, v in data['mapping'][roi_class]['id_to_roi'].items()}

    for hem in ['lh', 'rh']:
        data['fsaverage'][hem]['all-vertices'] = np.load(os.path.join(roi_dir, f'{hem}.all-vertices_fsaverage_space.npy'))
        for roi_class in CLASS_TO_ROI.keys():
            data['challenge'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_challenge_space.npy'))
            data['fsaverage'][hem][roi_class] = np.load(os.path.join(roi_dir, f'{hem}.{roi_class}_fsaverage_space.npy'))
    return data
