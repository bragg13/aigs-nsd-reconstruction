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
    return img[:,:,0]+img[:,:,1]*256+img[:,:,2]*(256**2)

def maskToUniqueIndices(img):
    imgSegIds = np.unique(maskToIndices(img)) # returns a sorted array of unique segment IDs found in the image
    imgSegIds = imgSegIds[imgSegIds != 0]
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

# for i in range(1,2): 
for i in range(len(cocoId_arr)): 
    cId = cocoId_arr[i]
    png_name = imgDir + '%012d.png' % cId

    crop = nsdcrop_arr[i]
    img = skimage.io.imread(png_name)
    croppedImg = resize(applyCropToImg(img, crop), (minSize,minSize), order=0)

    # croppedImg = (croppedImg * 256.).astype('uint8') # this we dont need because croppedImg is in the range 0-255

    imgSegIds = maskToUniqueIndices(croppedImg.astype('uint32'))
    catIds = getCategoryIDs(imgIdToAnns[cId], imgSegIds)
    
    catNames = getCategoryNames(catIdToCat, catIds)
    categories[cId] = catNames

    # print (f'cat ids: {catIds}')
    # print (f'cat names: {catNames}')
    # print (getSupercategoryNames(catIdToCat, catIds))

# %% convert category dict to dataframe
def getCategoriesForImg(cocoId):
    print(categories[cocoId])
    return categories[cocoId]

getCategoriesForImg(262239)