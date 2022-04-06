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

name_list = ['DR']
for data_name in name_list:
    
    counter = 0
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1') 
    
    data_type1 = 'pixel'
    data_type2 = 'dct'
    data_type3 = 'dwt'
    
    weight_path =save_path +data_type1+'_'+data_type2+'_'+data_type3+'/'+data_name+'/mlp_weight1/'
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
    # trainS1 = my_norm(trainS1)
    # trainS2 = my_norm(trainS2)
    # trainS3 = my_norm(trainS3)
    # testS1 = my_norm(testS1)
    # testS2 = my_norm(testS2)
    # testS3 = my_norm(testS3)
    trainS = np.concatenate((trainS1,trainS2,trainS3),axis = 1)
    testS = np.concatenate((testS1,testS2,testS3),axis = 1)
    # labelTr = np.argmax((labelTr),axis = 1)
    # labelTs = np.argmax((labelTs),axis = 1)
    del trainS1,trainS2,trainS3,testS1,testS2,testS3
    
    weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
    classes = list(range(0,max(labelTr)+1))
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    #%%
    seeds = np.random.randint(0,1000,10)#[308,951,14,619,764,702,722,616,570,597]

    for it in range(0,10):
    
        seed = int(seeds[it])
        reset_random_seeds(seed)
        model = denseMlpCreate_comb3(seed)
        savepath = weight_path + str(seed)+'/'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        #%% train
        checkpoint = ModelCheckpoint(savepath+'/checkpoint.h5', monitor='acsa', save_best_only=True, mode='max', verbose = 1)
        earlystop = EarlyStopping(monitor='acsa', min_delta=0, patience=50, verbose=0, mode='max', baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystop]
        h = model.fit(x = trainS,y= to_categorical(labelTr), batch_size=batch_size, epochs=epochs, class_weight = class_weights, verbose=2, shuffle=True,callbacks = callbacks_list,validation_data = (testS,to_categorical(labelTs)))
        model_json = model.to_json()
        with open(savepath+'/model.json', "w") as json_file:
            json_file.write(model_json)
        with open(savepath+'/model_history.json', 'w') as f:
            json.dump(str(h.history), f)
        
        
        model.load_weights(savepath+'/checkpoint.h5')
        pLabel=np.argmax(model.predict(testS), axis=1)
        tpr, acsa, sdcsa, prec, acsp, sdcsp, f1score, acsf, sdcsf, conf = spp.indices(pLabel, labelTs)
        r = [it]+[seed] + list(tpr)+ [acsa]+ [sdcsa]+list(prec)+[acsp]+[sdcsp]+ list(f1score)+[acsf]+[sdcsf]
    
            
        for kk in range(0,len(r)):
            sheet1.write(counter+it, kk, round(r[kk],2)) 
    wb.save(weight_path+'result.xls') 
