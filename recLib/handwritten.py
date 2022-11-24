# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from .utils import padToHeight,LOG_INFO,random_exec,create_dir,paper_noise
tqdm.pandas()
#--------------------
# helpers
#--------------------
def resizeToHeight(img,height):
    h,w=img.shape
    width=int((height*w)/h)
    img=cv2.resize(img,(width,height))
    return img

def createImgFromComps(df,comps):
    # get img_paths
    img_paths=[]
    for comp in comps:
        cdf=df.loc[df.label==comp]
        cdf=cdf.sample(frac=1)
        cdf.reset_index(drop=True,inplace=True)
        img_paths.append(cdf.iloc[0,2])
    
    # get images
    imgs=[cv2.imread(img_path,0) for img_path in img_paths]
    # get heights
    img_heights=[img.shape[0] for img in imgs]
    # distribution of 60-40
    if random_exec():
        dim_height=random.choice(img_heights)
        correct_height=resizeToHeight
    else:
        dim_height=max(img_heights)
        correct_height=padToHeight
        
    cimgs=[]
    for img in imgs:
        height=img.shape[0]
        if height!=dim_height:img=correct_height(img,dim_height)
        cimgs.append(img)
    img=np.concatenate(cimgs,axis=1)        
    return img


    
#--------------------
# ops
#--------------------
def createSyntheticData(save,
                        comp_df,
                        dictionary,
                        fiden=0):
    # dataframe vars
    for idx in tqdm(range(len(dictionary))):
        try:
            text=dictionary.iloc[idx,0]
            dtype=dictionary.iloc[idx,1]
            # image
            img=createImgFromComps(df=comp_df,comps=list(text))
            img=255-img
            img=paper_noise(img)
            # save
            fname=f"{fiden}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{os.path.join(save.img,fname)}\t{text}\t{dtype}\n")
        except Exception as e:
           LOG_INFO(e)