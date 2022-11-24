#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import pandas as pd 
from tqdm import tqdm

from recLib.utils import create_dir,LOG_INFO
from recLib.handwritten import createSyntheticData
from recLib.dataset import DataSet
from multiprocessing import Process
tqdm.pandas()
#------------------------
# fixed
#-------------------------
SPLIT=10240
#--------------------
# main
#--------------------
def main(args):
    iden        =   args.iden
    dict_csv    =   args.dict_csv
    data_dir    =   args.data_dir
    save_dir    =   args.save_path
    num_proc    =   int(args.num_process)

    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    LOG_INFO(save_dir)
    # save_paths
    class save:    
        img=create_dir(save_dir,"images")
        txt=os.path.join(save_dir,"data.txt")
    
    ds=DataSet(data_dir)
    df=pd.read_csv(dict_csv)
    dfs=[df[idx:idx+SPLIT] for idx in range(0,len(df),SPLIT)]
    max_end=len(dfs)

    def run(idx):
        if idx <len(dfs):
            createSyntheticData(save,ds.df,dfs[idx],idx*SPLIT)

    def execute(start,end):
        process_list=[]
        for idx in range(start,end):
            p =  Process(target= run, args = [idx])
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()


    if max_end==1:
        dfs=[df]
        run(0)
    elif max_end<=num_proc:
        for i in range(0,max_end):
            start=i
            end=start+max_end-1
            execute(start,end) 
    else:
        for i in range(0,max_end,num_proc):
            start=i
            end=start+num_proc
            if end>max_end:end=max_end-1
            execute(start,end) 
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Synthetic Dataset Creating Script")
    parser.add_argument("iden",help="identifier to identify the dataset")
    parser.add_argument("dict_csv", help="Path of the csv that contains dict info")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--num_process",required=False,default=16,help ="number of processes to be used:default=16")
    args = parser.parse_args()
    main(args)