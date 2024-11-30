

# %%
# show smth
# def visualisation_brain_and_image(img=0, hemisphere='left', roi='V1v'):
#     hemispheres = ['left', 'right']
#     roi = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]

#     # Load the image
#     img_dir = os.path.join(train_img_dir, train_img_list[img])
#     train_img = Image.open(img_dir).convert('RGB')

#     # Plot the image
#     plt.figure()
#     plt.axis('off')
#     plt.imshow(train_img)
#     plt.title('Training image: ' + str(img+1));
#     plt.show()


#     print(f"fsav shape {fsaverage_response.shape}")
#     # Create the interactive brain surface map
#     fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
#     view = plotting.view_surf(
#         surf_mesh=fsaverage['infl_'+hemisphere],
#         surf_map=fsaverage_response,
#         bg_map=fsaverage['sulc_'+hemisphere],
#         threshold=1e-14,
#         cmap='cold_hot',
#         colorbar=True,
#         title=roi+', '+hemisphere+' hemisphere'
#         )
#     return view

# %% Visualise the brain surface map and the image
# view = visualisation_brain_and_image(img=img, hemisphere='left', roi='V1v')
# view.open_in_browser()

# %%
# imports
import os
import numpy as np
from nilearn import datasets, plotting

from matplotlib import pyplot as plt
from nilearn.surface import load_surf_mesh

# %%
def get_roi_class(roi):
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


# %%
# download the dataset
def get_fmri_vector():
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
    return lh_fmri, rh_fmri
# %%
def get_fmri_data_roi(subj=3, roi='V1v', hemisphere='lh'):
    data_dir = os.path.join('./data', 'subj0'+str(subj))
    hemisphere += '.'

    roi_class = get_roi_class(roi)

    # Load the ROI brain surface maps
    challenge_roi_class_dir = os.path.join(data_dir, 'roi_masks',
        hemisphere+roi_class+'_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(data_dir, 'roi_masks',
        hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
    roi_map_dir = os.path.join(data_dir, 'roi_masks',
        'mapping_'+roi_class+'.npy')

    challenge_roi_class = np.load(challenge_roi_class_dir)
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    print(f"roi mapping: {roi_mapping}")
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)
    print(f"roi challenge: {len(challenge_roi)}, {type(challenge_roi)}")
    print(f"roi fsaverage: {len(fsaverage_roi)}, {type(fsaverage_roi)}")

    return challenge_roi, fsaverage_roi

# %%
def map_fmri_on_brain_surface(challenge_roi, fsaverage_roi, hemisphere, lh_fmri, rh_fmri, img_index=0):
    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_roi))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_roi)[0]] = lh_fmri[img_index,np.where(challenge_roi)[0]]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_roi)[0]] = rh_fmri[img_index,np.where(challenge_roi)[0]]
    return fsaverage_response

# %%
def get_surface_mesh(fsaverage_response, hemisphere='left'):
    # get coordinates and faces for fsaverage
    side = f"infl_{hemisphere}"
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage") [side]
    coords, faces = load_surf_mesh(fsaverage)

    # mask the vertices with non-zero BOLD signal
    response_mask = np.where(fsaverage_response)[0]

    # filter the coordinates from fsaverage mesh
    filtered_coords = coords[response_mask]

    # map from old vertex indices to new ones
    index_mapping = np.full(np.max(faces) + 1, -1)  # -1 is the inital value
    index_mapping[response_mask] = np.arange(response_mask.size)

    # adjust faces to new indexing and filter out invalid faces
    filtered_faces = index_mapping[faces]
    valid_faces_mask = np.all(filtered_faces != -1, axis=1)
    filtered_faces = filtered_faces[valid_faces_mask]

    # return the filtered coordinates, corresponding BOLD signal values, and adjusted faces
    return filtered_coords, fsaverage_response[response_mask], filtered_faces

# %%
# actual execution
lh_fmri, rh_fmri = get_fmri_vector()
coords_faces = {}

using_roi = ["EBA", "FBA-1", "FBA-2"]# this has no response, "mTL-bodies"]
for roi in using_roi:
    # lh
    challenge, fsaverage = get_fmri_data_roi(subj=3, roi=roi, hemisphere='lh')
    response = map_fmri_on_brain_surface(challenge, fsaverage, 'left', lh_fmri, rh_fmri, img_index=0)
    filtered_coords, masked_response, filtered_faces = get_surface_mesh(response, hemisphere='left')
    coords_faces[roi] = {'lh': {}, 'rh': {}}

    if filtered_coords.shape[0] > 0:
        coords_faces[roi]['lh']['coords'] = filtered_coords
        coords_faces[roi]['lh']['faces'] = filtered_faces
        coords_faces[roi]['lh']['response'] = masked_response

    # rh
    challenge, fsaverage = get_fmri_data_roi(subj=3, roi=roi, hemisphere='rh')
    response = map_fmri_on_brain_surface(challenge, fsaverage, 'right', lh_fmri, rh_fmri, img_index=0)
    filtered_coords, masked_response, filtered_faces = get_surface_mesh(response, hemisphere='right')
    if filtered_coords.shape[0] > 0:
        coords_faces[roi]['rh']['coords'] = filtered_coords
        coords_faces[roi]['rh']['faces'] = filtered_faces
        coords_faces[roi]['rh']['response'] = masked_response
        print(f"roi: {roi} - hemisphere: rh - coords: {filtered_coords.shape} - faces: {filtered_faces.shape} - response: {masked_response.shape}")


# %%
#
# plot the surface mesh in 3D with BOLD signal values using plt
# create a 3D figure
fig = plt.figure(figsize=(10, 10))
for i in using_roi:
    ax = fig.add_subplot(2, 2, using_roi.index(i)+1, projection='3d')
    coords_lh, coords_rh = coords_faces[i]['lh']['coords'], coords_faces[i]['rh']['coords']
    faces_lh, faces_rh = coords_faces[i]['lh']['faces'], coords_faces[i]['rh']['faces']
    res_lh, res_rh = coords_faces[i]['lh']['response'], coords_faces[i]['rh']['response']

    ax.plot_trisurf(coords_lh[:, 0], coords_lh[:, 1], coords_lh[:, 2], triangles=faces_lh,  linewidth=0.2)
    ax.plot_trisurf(coords_rh[:, 0], coords_rh[:, 1], coords_rh[:, 2], triangles=faces_rh,  linewidth=0.2)
    ax.set_title('BOLD signal on brain surface - ' + i)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # change view angle
    ax.view_init(10, 90)

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.set_title('BOLD signal on brain surface - ' + i)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# change view angle
ax.view_init(10, 90)

for i in using_roi:
    coords_lh, coords_rh = coords_faces[i]['lh']['coords'], coords_faces[i]['rh']['coords']
    faces_lh, faces_rh = coords_faces[i]['lh']['faces'], coords_faces[i]['rh']['faces']
    res_lh, res_rh = coords_faces[i]['lh']['response'], coords_faces[i]['rh']['response']

    ax.plot_trisurf(coords_lh[:, 0], coords_lh[:, 1], coords_lh[:, 2], triangles=faces_lh,  linewidth=0.2)
    ax.plot_trisurf(coords_rh[:, 0], coords_rh[:, 1], coords_rh[:, 2], triangles=faces_rh,  linewidth=0.2)

# ax = fig.add_subplot(2, 2, len(coords)+1, projection='3d')
# ax.plot_trisurf(coords_lh[:, 0], coords_lh[:, 1], coords_lh[:, 2], triangles=faces_lh, facecolors=plt.cm.viridis(res_lh), linewidth=0.2)
# ax.plot_trisurf(coords_rh[:, 0], coords_rh[:, 1], coords_rh[:, 2], triangles=faces_rh, facecolors=plt.cm.viridis(res_rh), linewidth=0.2)
# ax.set_title('BOLD signal on brain surface - all rois' )
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # change view angle
# ax.view_init(0, 90)

plt.show()

# %%
