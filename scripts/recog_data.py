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
from tqdm import tqdm
from recLib.utils import *
from recLib.handwritten import createSyntheticData
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):
    iden        =   args.iden
    dict_csv    =   args.dict_csv
    data_dir    =   args.data_dir
    save_path   =   args.save_path
    
    createSyntheticData(iden=iden,
                        save_dir=save_path,
                        data_dir=data_dir,
                        dict_csv=dict_csv)
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
    args = parser.parse_args()
    main(args)