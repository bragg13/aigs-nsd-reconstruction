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
# %% load nsd_coco dataframe
nsd_coco = read_and_preprocess()
