# %% coco load and imports
from pycocotools.coco import COCO
import skimage.io as io
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% loading the annotations for val and train
dataDir='..'
annFileVal=f'{dataDir}/annotations/instances_val2017.json'
annFileTrain=f'{dataDir}/annotations/instances_train2017.json'

# %%
coco_val2017=COCO(annFileVal)
coco_train2017=COCO(annFileTrain)

# %% dropping some columns so it's cleaner
def preprocess(dataDir='..'):
    useless_cols = ['Unnamed: 0', 'loss', 'flagged','BOLD5000',
        'subject1_rep0','subject1_rep1','subject1_rep2','subject2_rep0','subject2_rep1','subject2_rep2','subject3_rep0','subject3_rep1','subject3_rep2','subject4_rep0','subject4_rep1','subject4_rep2','subject5_rep0','subject5_rep1','subject5_rep2','subject6_rep0','subject6_rep1','subject6_rep2','subject7_rep0','subject7_rep1','subject7_rep2','subject8_rep0','subject8_rep1','subject8_rep2'
    ]
    nsd_coco = pd.read_csv(f'{dataDir}/nsd_coco.csv')
    nsd_coco.drop(columns=useless_cols, inplace=True)
    return nsd_coco

def filter_subj(subject):
    nsd_coco = preprocess()
    nsd_coco = nsd_coco[nsd_coco[f'subject{subject}'] == True]
    return nsd_coco[['nsdId','shared1000']]

# %% separate shared images from subject images
subject_cols = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8']
nsd_coco = preprocess(dataDir)

shared_images = nsd_coco[nsd_coco['shared1000'] == True]
shared_images = shared_images.drop(columns=['shared1000'])
subject_dfs = []
for i in range(1, 9):
    img_df = nsd_coco[(nsd_coco[f'subject{i}'] == True) & (nsd_coco['shared1000'] == False)]
    img_df = img_df.drop(columns=subject_cols)
    subject_dfs.append(img_df)
    print(f'subject{i}: {subject_dfs[i-1].shape[0]} images')
# %%
subject_dfs[0]

# %% function to get categories for each image
def get_categories():
    image_ids_val = coco_val2017.getImgIds()
    image_ids_train = coco_train2017.getImgIds()
    data = []

    for img_id in image_ids_val:
        # Get category IDs for the given image
        ann_ids = coco_val2017.getAnnIds(imgIds=img_id)
        anns = coco_val2017.loadAnns(ann_ids)
        category_ids = {ann['category_id'] for ann in anns}

        # Map image ID to category names
        category_names = [coco_val2017.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
        data.append({'cocoId': img_id, 'categories': str(category_names)})

    for img_id in image_ids_train:
            # Get category IDs for the given image
            ann_ids = coco_train2017.getAnnIds(imgIds=img_id)
            anns = coco_train2017.loadAnns(ann_ids)
            category_ids = {ann['category_id'] for ann in anns}

            # Map image ID to category names
            category_names = [coco_train2017.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
            data.append({'cocoId': img_id, 'categories': str(category_names)})
    return pd.DataFrame(data)
df = get_categories()
print(f'tot number of images: {len(df)}')

# %% add categories to nsd_coco and images
cat_df = get_categories()
nsd_coco = pd.merge(nsd_coco, cat_df, left_on='cocoId', right_on='cocoId', how='inner')
for i in range(1, 9):
    merged = pd.merge(subject_dfs[i-1], cat_df, left_on='cocoId', right_on='cocoId', how='inner')
    print(f"subj{i}: {subject_dfs[i-1].shape[0]} images, merged: {merged.shape[0]} images")
    subject_dfs[i-1] = merged

# %% syntactic sugar to retrieve categories from nsdId or cocoId
def getCategoryFromCocoId(cocoId):
    return nsd_coco[nsd_coco['cocoId'] == cocoId]['categories']

def getCategoryFromNsdId(nsdId):
    return nsd_coco[nsd_coco['nsdId'] == nsdId]['categories']

# %% test it out
# nsd_coco contains all the images
# images is an array of dataframes, each containing the images for a subject
# shared_images contains the images shared by all subjects
getCategoryFromNsdId(72976)
