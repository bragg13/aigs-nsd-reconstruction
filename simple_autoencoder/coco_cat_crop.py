# %%
import sys
import os
import struct
import time
import numpy as np
import h5py
from glob import glob
import scipy.io as sio
from scipy import ndimage as nd
from scipy import misc
from scipy.io import loadmat
from tqdm import tqdm
import multiprocessing as mp
import pickle
import math
import matplotlib.pyplot as plt
import PIL.Image as pim
import PIL.ImageOps as pop
# import seaborn as sns

fpX = np.float32

# %% not needed
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

# %% functions needed

# creates a unique idenifier for each unique color, base-256 presentation
# commonly used for image segmentation tasks
# returns array with shape (h,w)
def maskToIndices(img):
    return img[:,:,0] + (img[:,:,1] * 256) + (img[:,:,2] * (256**2))

def maskToUniqueIndices(img):
    imgSegIds = list(np.unique(maskToIndices(img)))
    if 0 in imgSegIds:
        imgSegIds.remove(0)
    return np.unique(imgSegIds) # Returns the sorted unique elements of an array

# imgSegIds get from maskToUniqueIndices
def getCategoryIDs(annotations, imgSegIds):
    segToCatId = defaultdict(list)
    for ann in annotations:
        for seg in ann['segments_info']:
            segToCatId[seg['id']] = seg['category_id']
    return [segToCatId[s] for s in imgSegIds if s in segToCatId]

def getCategoryNames(catIdToCat, catIds):
    # getCategoryNames(catToCat, catIds)
    return np.unique([catIdToCat[c][0]['name'] for c in catIds])

def getSupercategoryNames(catIdToCat, catIds):
    # getSupercategoryNames(catToCat, catIds)
    return np.unique([catIdToCat[c][0]['supercategory'] for c in catIds])

def supercategoryMap(croppedImg, annotations, embbeding):
    # supercategoryMap(croppedImg, imgIdToAnns[cId], class_embbeding)
    segmentMap = maskToIndices(croppedImg).flatten()
    superMap = np.full(fill_value=-1, shape=segmentMap.shape, dtype=np.int)
    imgSegIds = maskToUniqueIndices(croppedImg)
    catIds = getCategoryIDs(annotations, imgSegIds)
    for c,s in zip(catIds, imgSegIds):
        supercat = getSupercategoryNames(catIdToCat, [c])[0]
        superMap[segmentMap==s] = embbeding[supercat]            
    return superMap.reshape(croppedImg.shape[:2])

# 
def getCategoryIDs(annotations, imgSegIds):
    segToCatId = defaultdict(list)
    for ann in annotations:
        for seg in ann['segments_info']:
            segToCatId[seg['id']] = seg['category_id']
    return [segToCatId[s] for s in imgSegIds if s in segToCatId]

# %%
import json
from collections import defaultdict

annDir = '..'
imgDir = annDir + '/panoptic_annotations/' # combined folder with train2017 and val2017 png masks
panop_trn_annFile = imgDir + 'panoptic_train2017.json'
panop_val_annFile = imgDir + 'panoptic_val2017.json'

dataset = dict() 
dataset = json.load(open(panop_trn_annFile, 'r'))

imgIdToAnns = defaultdict(list) # has ann from train and val
catIdToCat = defaultdict(list) # has cat only from train

# %% prints 
print(dataset.keys()) # info, licenses, images, annotations, categories

print('\nannotations[0]:')
print(f'segments_info: {dataset["annotations"][0]["segments_info"]}')
print(f'file_name: {dataset["annotations"][0]["file_name"]}')
print(f'image_id: {dataset["annotations"][0]["image_id"]}')

print('\ncategories[0]:')
print(f'{dataset["categories"][0]}')

#%%
if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann) # annotations: sgements_info, file_name, image_id
    for cat in dataset['categories']:
        catIdToCat[cat['id']].append(cat)

# %%
dataset = dict()
dataset = json.load(open(panop_val_annFile, 'r'))

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)

# %%
import skimage.io
from skimage.transform import resize

minSize = 227
start_idx = 3
n_idx = 1
subject = 3

for i in range(n_idx): # for each subj do this
    nId = start_idx + i
    cId = cocoId_arr[subject-1, nId] # array of cocoIds per subj <- nsd_to_coco_indice_map.h5py
    png_name = imgDir + '%012d.png' % cId

    crop = nsdcrop_arr[subject-1, nId] # array of cropBox per subj <- nsd_to_coco_indice_map.h5py
    img = skimage.io.imread(png_name)

    croppedImg = applyCropToImg(img, crop) # img is a 3d or 4d array
    croppedImg = (resize(croppedImg, (minSize,minSize), order=0) * 256.).astype('uint8')

    # use below
    imgSegIds = maskToUniqueIndices(croppedImg) # unique segment IDs, croppedImg is a 3d or 4d array
    catIds = getCategoryIDs(imgIdToAnns[cId], imgSegIds)

    print (catIds)
    print (getCategoryNames(catIdToCat, catIds))
    print (getSupercategoryNames(catIdToCat, catIds))