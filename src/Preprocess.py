import os 
import warnings 
warnings.filterwarnings('ignore')
from glob import glob 
from tqdm import tqdm 

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

import torch 
import torch.nn as nn  
from torch import Tensor
from torch.utils.data import Dataset,DataLoader 

def df_columns(df):
    columns = df.iloc[0,:].values
    df.columns = columns 
    df = df.iloc[1:,:].reset_index(drop=True)
    df.columns = list(pd.Series(df.columns).apply(lambda x : x.split(' ')[-1]).values)
    return df 

def df_type(df):
    data = df.iloc[:,1:-1].astype(float)
    
    columns = list(df.columns)
    columns.remove('Timestamp')
    columns.remove('Normal/Attack')

    df[columns] = data 
    return df 

class Normalize:
    def __init__(self,train,test):
        self.train = train.copy()
        self.test = test.copy()
        self.target = self.target_columns(train)
        self.train_min = np.min(self.train[self.target])
        self.train_max = np.max(self.train[self.target])

    def target_columns(self,df):
        columns = list(df.columns)
        columns.remove('Timestamp')
        columns.remove('Normal/Attack')
        return columns         
        
    def scaling(self,seq):
        return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    
    
    def __call__(self):
        
        self.train[self.target] =  self.scaling(self.train[self.target])
        self.test[self.target]  =  self.scaling(self.test[self.target])      
        
        #1 값만 있는 경우 스케일링 후 null이 되므로 1로 다 채움 
        self.train = self.train.fillna(1)
        self.test = self.test.fillna(1)
        
        return self.train,self.test 
    
def split_window(df:pd.DataFrame,window_size=320,step=1):
    labels = df['Normal/Attack']
    df = df.drop(columns = ['Timestamp','Normal/Attack']).values
    
    length = (df.shape[0] - window_size) // step + 1
    
    df_window = [] 
    label_window = [] 
    left = 140 
    right = window_size - left 
    for i in tqdm(range(left//step,length)):
        time_window = np.float32(df[i*step-left:i*step+right]) #140 - x - 180 
        time_labels = labels[i*step-left:i*step+right]
        
        label_window.append(time_labels)
        df_window.append(time_window)        
    return np.array(df_window),np.array(label_window)
       

def df_preprocess(df):
    #컬럼 
    df = df_columns(df)
    #타입 
    df = df_type(df)
    #시간정렬
    df = df.sort_values(by='Timestamp').reset_index(drop=True) 
    #라벨링
    df['Normal/Attack'] = df['Normal/Attack'].apply(lambda x : 0 if x=='Normal' else 1)
    return df 

def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[j,:] = x[i,:]
            j += 1
            x_1[j,:] = (x[i,:] + x[i+1, :]) /2 
            j += 1
        else:
            x_1[j,:] = x[i,:]
            j += 1
    return x_1    
    
def data_aug(train_x,train_y,times=2):
    res_train_x=[]
    res_train_y=[]
    for idx in tqdm(range(train_x.shape[0])):
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug=aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)

    res_train_x=np.array(res_train_x)
    res_train_y=np.array(res_train_y)

    return res_train_x,res_train_y