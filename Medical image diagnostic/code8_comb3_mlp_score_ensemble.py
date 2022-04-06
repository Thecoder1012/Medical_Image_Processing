# import packages
from numpy import loadtxt
import numpy as np
import random
from my_metrics import average_class_specific_accuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import json
from tensorflow.python.keras import backend as K
import os
from plot_confusion_mat import *
from sklearn.utils import class_weight
from my_resnet import *
import json 
import matplotlib.pyplot as plt
import xlwt 
from xlwt import Workbook
from load_data import load_intermediate_data
import ast
from sklearn.model_selection import train_test_split
import suppli as spp
# %% set paths and parameters

save_path = '' # add the base path to save model and intermediate data

name_list = ['BHI']
for data_name in name_list:
    
    counter = 0
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1') 
    
    data_type1 = 'pixel'
    data_type2 = 'dct'
    data_type3 = 'dwt'
    
    weight_path1 = save_path +data_type1+'/'+data_name+'/mlp_weight/'
    weight_path2 = save_path +data_type2+'/'+data_name+'/mlp_weight/'
    weight_path3 = save_path +data_type3+'/'+data_name+'/mlp_weight/'
    weight_path =save_path +data_type1+'_'+data_type2+'_'+data_type3+'/'+data_name+'/mlp_weight_single_combined/'

    # intermediate_datafolder1 = 'D:/Simpi/Work/COVID19/19082021/output/'+data_type1+'_intermediate/'+data_name+'/'
    # intermediate_datafolder2 = 'D:/Simpi/Work/COVID19/19082021/output/'+data_type2+'_intermediate/'+data_name+'/'
    # intermediate_datafolder3 = 'D:/Simpi/Work/COVID19/19082021/output/'+data_type3+'_intermediate/'+data_name+'/'
    
    batch_size = 32 
    epochs = 300
    input_size = 224
    
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    
    #%%
    def reset_random_seeds(seed):
       os.environ['PYTHONHASHSEED']=str(seed)
       os.environ['TF_DETERMINISTIC_OPS'] = '1'
       tf.random.set_seed(seed)
       np.random.seed(seed)
       random.seed(seed)
             
    def my_norm(x):
        return  (x-x.min(axis = 1,keepdims = True))/(x.max(axis = 1,keepdims = True)-x.min(axis = 1,keepdims=True))
    
    #%% load data and train and save
    trainS1,labelTr, testS1, labelTs = load_intermediate_data(data_name,data_type1)
    trainS2,labelTr, testS2, labelTs = load_intermediate_data(data_name,data_type2)
    trainS3,labelTr, testS3, labelTs = load_intermediate_data(data_name,data_type3)
    # labelTr = np.argmax((labelTr),axis = 1)
    # labelTs = np.argmax((labelTs),axis = 1)
    
    weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
    classes = list(range(0,max(labelTr)+1))
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    #%%
    seeds = np.random.randint(0,1000,10)#[308,951,14,619,764,702,722,616,570,597]

    model_list1 = os.listdir(weight_path1)
    model_list2 = os.listdir(weight_path2)
    model_list3 = os.listdir(weight_path3)  
    seed = 0
    model = denseMlpCreate(seed)
    rnge = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for i in rnge:
        for j in rnge:
            for k in rnge:
                if round(i+j+k,1) == 1.0:
                    r = np.zeros((10,6))
                    for it in range(0,10):
                        print(i,j,k,it)
                    
                        model.load_weights(weight_path1+'/'+model_list1[it]+'/'+'/checkpoint.h5')
                        pLabel1 = model.predict(testS1)
                    
                        model.load_weights(weight_path2+'/'+model_list2[it]+'/'+'/checkpoint.h5')
                        pLabel2 = model.predict(testS2) 
                        
                        model.load_weights(weight_path3+'/'+model_list3[it]+'/'+'/checkpoint.h5')
                        pLabel3 = model.predict(testS3)
                        
                        pLabel = np.argmax(i*pLabel1+j*pLabel2+k*pLabel3, axis=1)
                        _,  r[it,0],  r[it,1], _,  r[it,2],  r[it,3],  _,  r[it,4],  r[it,5], _ = spp.indices(pLabel, labelTs)
                    rr = np.mean(r,axis = 0)
                    sr = np.std(r,axis = 0)
                    rr = np.append([i,j,k],rr)
                    rr = np.append(rr,sr)

                    counter = counter+1
                    for kk in range(0,len(rr)):
                        sheet1.write(counter, kk, round(rr[kk],2)) 
    wb.save(weight_path+'result.xls') 
