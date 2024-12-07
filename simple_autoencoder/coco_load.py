# %% imports in use
# from pycocotools.coco import COCO
from logger import log
import pandas as pd
from coco_cat import extract_categories
debug = False

# %% dropping unnecessary columns, and adding Coco categories to nsd-coco dataframe
dataDir='..'

def read_and_preprocess(dataDir='..'):
    useless_cols = ['Unnamed: 0', 'loss', 'flagged','BOLD5000',
        'subject1_rep0','subject1_rep1','subject1_rep2','subject2_rep0','subject2_rep1','subject2_rep2','subject3_rep0','subject3_rep1','subject3_rep2','subject4_rep0','subject4_rep1','subject4_rep2','subject5_rep0','subject5_rep1','subject5_rep2','subject6_rep0','subject6_rep1','subject6_rep2','subject7_rep0','subject7_rep1','subject7_rep2','subject8_rep0','subject8_rep1','subject8_rep2'
    ]
    nsd_coco = pd.read_csv(f'{dataDir}/nsd_coco.csv')
    nsd_coco.drop(columns=useless_cols, inplace=True)

    log(f'nsd-coco loaded: {len(nsd_coco)} images', 'COCO_LOAD')
    return nsd_coco

# nsd_coco = read_and_preprocess() # for cell based testing

# %% retrieve shared images from nsd_coco (incl categories)
subject_cols = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8']

def shared_imgs_df(nsd_coco):
    shared = nsd_coco[nsd_coco['shared1000'] == True]
    shared = shared.drop(columns=['shared1000'])
    shared = shared.drop(columns=subject_cols)
    if debug:
        print(f'\ncolumns: {shared.columns}')
        print(f'shared: {len(shared)} images')
    return shared

# shared_df = shared_imgs_df() # for cell based testing

# %% retrieve subject specific images from nsd_coco (incl categories)
def subject_dfs(nsd_coco):
    subject_dfs = []
    if debug: print(f'\nsubj_dfs:')
    for i in range(1, 9):
        img_df = nsd_coco[(nsd_coco[f'subject{i}'] == True) & (nsd_coco['shared1000'] == False)]
        img_df = img_df.drop(columns=subject_cols)
        subject_dfs.append(img_df)
        if debug: print(f'subject{i}: {subject_dfs[i-1].shape[0]} images')
    return subject_dfs

# subj_dfs = subject_dfs() # for cell based testing

# %% get categories and filter by category
def get_categories_df(df):
    return pd.DataFrame.from_dict(extract_categories(df), orient='index', columns=['cocoId','categories'])

def filterByCategory(df, category: str, contain = True):
    if contain : df = df[df['categories'].apply(lambda x: category in x)]
    else : df =  df[df['categories'].apply(lambda x: category not in x)]
    # print(f'\n{category}: {len(df)} images')
    return df

def splitByCategory(df, category: str):
    df1 = df[df['categories'].apply(lambda x: category in x)]
    df2 = df[df['categories'].apply(lambda x: category not in x)]
    if debug:
        print(f'\ncategory split:')
        print(f'{category}: {len(df1)} images')
        print(f'not {category}: {len(df2)} images')
    return df1, df2

# %% syntactic sugar to retrieve categories from nsdId or cocoId
def getCategoryFromCocoId(nsd_coco, cocoId):
    return nsd_coco[nsd_coco['cocoId'] == cocoId]['categories']

def getCategoryFromNsdId(nsd_coco, nsdId):
    return nsd_coco[nsd_coco['nsdId'] == nsdId]['categories']

def getSharedDf(nsd_coco):
    shared = shared_imgs_df(nsd_coco)
    categories = get_categories_df(shared)
    return shared.merge(categories, on='cocoId')

def getSubjDf(nsd_coco, subj_index):
    return getSubjDfs(nsd_coco)[subj_index -1]

def getSubCatjDf(nsd_coco, subj_index):
    subj = getSubjDf(nsd_coco, subj_index)
    categories = get_categories_df(subj)
    return subj.merge(categories, on='cocoId')

def getSubjDfs(nsd_coco):
    return subject_dfs(nsd_coco)

# %% test it out
# nsd_coco contains all the images incl categories
# shared_df contains the images shared by all subjects
# subj_dfs is an array of dataframes, each containing the images for a subject

nsd_coco = read_and_preprocess()

# %% old loading the annotations for val and train
# annFileVal=f'{dataDir}/annotations/instances_val2017.json'
# annFileTrain=f'{dataDir}/annotations/instances_train2017.json'

# coco_val2017=COCO(annFileVal)
# coco_train2017=COCO(annFileTrain)

# %% old / function to get categories for each image
# def get_categories():
#     image_ids_val = coco_val2017.getImgIds()
#     image_ids_train = coco_train2017.getImgIds()
#     data = []

#     for img_id in image_ids_val:
#         # Get category IDs for the given image
#         ann_ids = coco_val2017.getAnnIds(imgIds=img_id)
#         anns = coco_val2017.loadAnns(ann_ids)
#         category_ids = {ann['category_id'] for ann in anns}

#         # Map image ID to category names
#         category_names = [coco_val2017.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
#         data.append({'cocoId': img_id, 'categories': str(category_names)})

#     for img_id in image_ids_train:
#         # Get category IDs for the given image
#         ann_ids = coco_train2017.getAnnIds(imgIds=img_id)
#         anns = coco_train2017.loadAnns(ann_ids)
#         category_ids = {ann['category_id'] for ann in anns}

#         # Map image ID to category names
#         category_names = [coco_train2017.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
#         data.append({'cocoId': img_id, 'categories': str(category_names)})

#     df = pd.DataFrame(data)
#     # print(f'\ntot number of images: {len(df)}')
#     return df

# %% old / function to merge coco categories into a dataframe
# def merge_categories(df):
#     return pd.merge(df, get_categories(), left_on='cocoId', right_on='cocoId', how='inner')
