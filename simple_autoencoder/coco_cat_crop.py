# %%
import skimage.io
from skimage.transform import resize
import numpy as np
import json
from collections import defaultdict
import ast
import coco_load as cl
# import seaborn as sns

# %% og functions taken from notebook
def applyCropToImg(img, box):
    '''
    applyCropToImg(img, cropBox)
    img ~ any h x w x n image
    cropBox ~ (top, bottom, left, right) in fractions of image size
    '''
    if box[0]+box[1] >= 1:
        raise ValueError('top and bottom crop must sum to less than 1')
    if box[2]+box[3] >= 1:
        raise ValueError('left and right crop must sum to less than 1')
    shape = img.shape
    topCrop = np.round(shape[0]*box[0]).astype(int)
    bottomCrop = np.round(shape[0]*box[1]).astype(int)
    leftCrop = np.round(shape[1]*box[2]).astype(int)
    rightCrop = np.round(shape[1]*box[3]).astype(int)
    croppedImage = img[topCrop:(shape[0]-bottomCrop),leftCrop:(shape[1]-rightCrop)]
    return croppedImage

# base-256 representation, commonly used for image segmentation tasks
# The resulting 2D array has a shape of h × w, where each element represents a unique integer ID corresponding to the color at that pixel
def maskToIndices(img):
    return img[:,:,0] + (img[:,:,1] * 256) + (img[:,:,2] * (256**2))

def maskToUniqueIndices(img):
    imgSegIds = np.unique(maskToIndices(img)) # returns a sorted array of unique segment IDs found in the image
    if 0 in list(imgSegIds): # 0 is probably black aka no segment/background, therefore has no category
        imgSegIds.remove(0)
    return np.unique(imgSegIds)

def getCategoryIDs(annotations, imgSegIds): # get imgSegIds with maskToUniqueIndices()
    segToCatId = defaultdict(list)
    for ann in annotations:
        for seg in ann['segments_info']:
            segToCatId[seg['id']] = seg['category_id']
    return [segToCatId[s] for s in imgSegIds if s in segToCatId]

def getCategoryNames(catIdToCat, catIds):
    return np.unique([catIdToCat[c][0]['name'] for c in catIds])

def getSupercategoryNames(catIdToCat, catIds):
    return np.unique([catIdToCat[c][0]['supercategory'] for c in catIds])

def supercategoryMap(croppedImg, annotations, embbeding):
    segmentMap = maskToIndices(croppedImg).flatten()
    superMap = np.full(fill_value=-1, shape=segmentMap.shape, dtype=np.int)
    imgSegIds = maskToUniqueIndices(croppedImg)
    catIds = getCategoryIDs(annotations, imgSegIds)
    for c,s in zip(catIds, imgSegIds):
        supercat = getSupercategoryNames(catIdToCat, [c])[0]
        superMap[segmentMap==s] = embbeding[supercat]            
    return superMap.reshape(croppedImg.shape[:2])

# %% directories
# in the panotic annotations download, there are 2 zip files that hold the images, join them into one folder 'panoptic_joint'
annDir = '../panoptic_annotations/'
imgDir = annDir + '/panoptic_joint/' # combined folder with train2017 and val2017 png masks
panop_trn_annFile = annDir + 'panoptic_train2017.json'
panop_val_annFile = annDir + 'panoptic_val2017.json'

# %% load annotations data
imgIdToAnns = defaultdict(list) # will have ann from train and val
catIdToCat = defaultdict(list) # will have cat from train (and val? see comment below)

dataset = dict() 
dataset = json.load(open(panop_trn_annFile, 'r'))

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann) # annotations: sgements_info, file_name, image_id
    for cat in dataset['categories']:
        catIdToCat[cat['id']].append(cat)

dataset = dict()
dataset = json.load(open(panop_val_annFile, 'r'))

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)
    # the below was originally not in the code, do we need it??
    for cat in dataset['categories']:
        catIdToCat[cat['id']].append(cat)

# %% prints to study dataset
# print(f'loaded dataset: {dataset.keys()}') # info, licenses, images, annotations, categories

# print('\nannotations[0]:')
# print(f'segments_info: {dataset["annotations"][0]["segments_info"]}')
# print(f'file_name: {dataset["annotations"][0]["file_name"]}')
# print(f'image_id: {dataset["annotations"][0]["image_id"]}')

# print('\ncategories[0]:')
# print(f'{dataset["categories"][0]}')

# %% load relevant coco ids and their crop dimensions from our coco_load dataframe
shared_df = cl.getSharedDf(cl.nsd_coco)
cocoId_arr = np.copy(shared_df['cocoId'].values)
nsdcrop_arr = shared_df['cropBox'].values # values are the cropBoxes but as strings
nsdcrop_arr = [ast.literal_eval(item) for item in nsdcrop_arr] # ast converts strings to tuples of floats

# %% get categories
minSize = 227 # from og code
categories = dict()

for i in range(len(cocoId_arr)): 
    cId = cocoId_arr[i]
    png_name = imgDir + '%012d.png' % cId

    crop = nsdcrop_arr[i]
    img = skimage.io.imread(png_name)

    croppedImg = applyCropToImg(img, crop)
    # debugging missing from here
    croppedImg = (resize(croppedImg, (minSize,minSize), order=0) * 256.).astype('uint8') # uint8 causes an error

    imgSegIds = maskToUniqueIndices(croppedImg)
    catIds = getCategoryIDs(imgIdToAnns[cId], imgSegIds)
    catNames = getCategoryNames(catIdToCat, catIds)
    
    categories[cId] = catNames

    print (catIds)
    print (getCategoryNames(catIdToCat, catIds))
    print (getSupercategoryNames(catIdToCat, catIds))

# %% convert category dict to dataframe
# %% copy from notebook 
# minSize = 227
# start_idx = 3
# n_idx = 1
# subject = 3

# # instead of below we can get the coco ids and crops from our dataframe
# coco_info_set = h5py.File(stim_root + 'nsd_to_coco_indice_map.h5py', 'r')
# cocoId_arr = np.copy(coco_info_set['cocoId'])
# nsdcrop_arr = np.copy(coco_info_set['crop'])

# # get the categories
# for i in range(n_idx): # for each image do this

#     nId = start_idx + i
#     cId = cocoId_arr[subject-1, nId] # array of cocoIds per subj <- nsd_to_coco_indice_map.h5py
#     png_name = imgDir + '%012d.png' % cId

#     crop = nsdcrop_arr[subject-1, nId] # array of cropBox per subj <- nsd_to_coco_indice_map.h5py
#     img = skimage.io.imread(png_name)

#     croppedImg = applyCropToImg(img, crop) # img is a 3d array
#     croppedImg = (resize(croppedImg, (minSize,minSize), order=0) * 256.).astype('uint8')

#     imgSegIds = maskToUniqueIndices(croppedImg) # unique segment IDs, croppedImg is a 3d array
#     catIds = getCategoryIDs(imgIdToAnns[cId], imgSegIds)

#     print (catIds)
#     print (getCategoryNames(catIdToCat, catIds))
#     print (getSupercategoryNames(catIdToCat, catIds))