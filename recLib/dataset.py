# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd 
from tqdm import tqdm
from .utils import LOG_INFO
tqdm.pandas()
#--------------------
# class info
#--------------------

        
class DataSet(object):
    def __init__(self,data_dir,allowed_symbols=["-","=","/"]):
        self.data_dir       =   data_dir
        
        class resources:
            class bangla:
                dir   =   os.path.join(data_dir,"bangla","numbers")
                csv   =   os.path.join(data_dir,"bangla","numbers.csv")
            class english:
                dir   =   os.path.join(data_dir,"english","numbers")
                csv   =   os.path.join(data_dir,"english","numbers.csv")
            class symbols:
                dir   =   os.path.join(data_dir,"common","symbols")
                csv   =   os.path.join(data_dir,"common","symbols.csv")

        self.resources=resources
        # get df
        self.resources.bangla.df       =  self.__getDataFrame(self.resources.bangla)
        self.resources.english.df      =  self.__getDataFrame(self.resources.english)
        self.resources.symbols.df      =  self.__getDataFrame(self.resources.symbols)
        # number dfs
        nums=[self.resources.bangla.df,self.resources.english.df]
        # symbols
        syms=[]
        for sym in allowed_symbols:
            _df=self.resources.symbols.df.loc[self.resources.symbols.df.label==sym]
            LOG_INFO(f"for {sym}:found {len(_df)}")
            _df.reset_index(drop=True,inplace=True)
            syms.append(_df)

        self.df             =pd.concat(nums+syms,ignore_index=False)

    

    def __getDataFrame(self,obj):
        '''
            creates the dataframe from a given csv file
            args:
                obj       =   the obj that has csv and dir
                
        '''
        try:
            df=pd.read_csv(obj.csv)
            assert "filename" in df.columns,f"filename column not found:{obj.csv}"
            assert "label" in df.columns,f"label column not found:{obj.csv}"
            df.label=df.label.progress_apply(lambda x: str(x))
            df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(obj.dir,f"{x}.bmp"))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{obj.csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 