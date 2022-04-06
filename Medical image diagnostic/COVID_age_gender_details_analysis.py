# import packages
from numpy import loadtxt
import numpy as np
import random
from my_metrics import average_class_specific_accuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_dct_dwt import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
# %% set paths and parameters
counter = 0
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1') 
pred_prob = []
true_cls = []
file_name = []

    
def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
    
        
def rotate(l, n):
    return l[n:] + l[:n]

def get_data_label(data_dir,text_file): 
    x = []
    y = []
    mapping={
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        x.append(np.load(data_dir + file_name+'.npy'))
        y.append(line_content[i].split(" ")[2].rstrip())
    y =list(map(mapping.get, y))

    return np.asarray(x,dtype = np.float32), to_categorical(np.asarray(y,dtype = np.int32))

for k in range(0,5):
    print('fold: ',k)
    txt_file_path = './chestxray1/txtfiles/'
    weight_path ='.COVID19/pixel_dct_dwt/fold'+str(k+1)+'/mlp_weight/'
    intermediate_datafolder = '.COVID19/pixel_intermediate_fold' +str(k+1)+'/'
    intermediate_dctfolder = '.COVID19/dct_intermediate_fold' +str(k+1)+'/'
    intermediate_dwtfolder = '.COVID19/dwt_intermediate_fold' +str(k+1)+'/'

    batch_size = 32 
    num_classes = 3
    epochs = 300
    input_size = 224
    no_fold = 5
    
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(intermediate_datafolder):
        os.makedirs(intermediate_datafolder)


    # load data and train and save
    fold_set = list(range(0,5))
    fold_set = rotate(fold_set, k)
    train_files = []
    test_fold = fold_set[0]
    val_fold = fold_set[1]
    train_fold = fold_set[2::]
    with open(txt_file_path+'fold_'+str(test_fold+1)+'.txt', 'r') as fr:
         test_files = fr.readlines()
    with open(txt_file_path+'fold_'+str(val_fold+1)+'.txt', 'r') as fr:
         val_files = fr.readlines()
    for i in train_fold:
         with open(txt_file_path+'fold_'+str(i+1)+'.txt', 'r') as fr:
             train_files.extend(fr.readlines())
    
        
    x_test_raw, y_test = get_data_label(intermediate_datafolder,test_files)        
    x_test_dct, y_test = get_data_label(intermediate_dctfolder,test_files)
    x_test_dwt, y_test = get_data_label(intermediate_dwtfolder,test_files)
        
    x_test = np.concatenate((np.squeeze(x_test_raw),np.squeeze(x_test_dct),np.squeeze(x_test_dwt)),axis = 1)

    testS = x_test
    labelTs = np.squeeze(y_test)
    # define model
    seed = 0
    model = denseMlpCreate_comb3(seed)
                
    model.load_weights(weight_path+'/checkpoint.h5')
    y_pred = model.predict(x = x_test)
    if k==0:
        pred_prob = y_pred 
        true_cls = labelTs 
    else:
        pred_prob = np.concatenate((pred_prob,y_pred),axis = 0)
        true_cls = np.concatenate((true_cls,labelTs),axis = 0)

    file_name = file_name + test_files
#         for kk in range(0,8):
#             sheet1.write(counter+it, kk, round(r[kk],2)) 
#     counter = counter + 12
# wb.save('C:/Users/susmi/for_paper/output/pixel_dct_dwt/'+'result.xls') 
#%%
import pickle

with open("./chestxray1/txtfiles/age_gender_details.txt", "rb") as fp:   #Pickling
    details_age= pickle.load(fp)
#%% agewise analysis
age =[]
true_class = []
pred_class = []
for i in range(0,len(file_name)):
    for j in range(0,len(details_age)):
       # print(i,j)
        if file_name[i].split(" ")[1]== details_age[j][0]:
            #print('i am here')
            if details_age[j][3] !=  '':
                print(details_age[j])
                age.append(int(details_age[j][3]))
                true_class.append(np.argmax(true_cls[i]))
                pred_class.append(np.argmax(pred_prob[i]))
                break

age_arr = np.asarray(age)
class_arr = np.asarray(true_class)
pred_arr = np.asarray(pred_class)
#%%
def give_counts(age_arr,class_arr, pred_arr,age1,age2,true_class, pred_class):
    a = np.where(np.logical_and(class_arr==true_class, pred_arr==pred_class))
    b = np.where(np.logical_and(age_arr>=age1, age_arr<age2))
    return np.intersect1d(a,b).size

#color = ['#aed6dc', '#ff9a8d', '#4a536b']
color1 = [  '#C1CD97','#E18D96','#78A2CC']
color2 = [  '#50874d','#A02C2D','#3065ac']
age_low_range = [0,20,40,60,80]
age_up_range = [20,40,60,80,100]
class_all = [0,1,2]
counts = np.zeros((3,3,5))
for k in class_all:
    for l in class_all:
        counter = 0
        for i,j in zip(age_low_range,age_up_range):
            counts[k,l,counter]=give_counts(age_arr,class_arr, pred_arr,i,j,k,l)
            counter = counter+1

counts_norm = np.zeros((3,3,5))
for k in class_all:
    for l in class_all:
        for j in [0,1,2,3,4]: 
            counts_norm[k,l,j] =  counts[k,l,j]/sum(counts[k,:,j])

fig, ax = plt.subplots()
pos = [0,1,2,3,4]
w = [-.3,0,.3]
for k in class_all:
    bottom = [0,0,0,0,0]
    for l in class_all:
        counter = 0
        pos_shift = [z + w[k] for z in pos]
        plt.bar(pos_shift, counts_norm[k,l,:], bottom = bottom, color=color1[l], edgecolor=color2[l], width=.25)
        bottom = [a + b for a, b in zip(bottom, list(counts_norm[k,l,:]))]

plt.legend(['Norm','Pne','COV'],loc='lower center',bbox_to_anchor=(0.50, -0.45),ncol = 3,title="Predicted class")
#plt.xticks(ticks = pos,labels = ['Male','Female'],position = (-0.2,-0.2))
plt.xticks(ticks = [-0.3,0,0.3,.7,1,1.3,1.7,2,2.3,2.7,3,3.3,3.7,4,4.3],labels = ['Norm','Pne','COV','Norm','Pne','COV','Norm','Pne','COV','Norm','Pne','COV','Norm','Pne','COV'],position = (-0.01,-0.01),rotation = 45)
plt.text(x = 0, y = -0.25,s ='0-20',horizontalalignment= 'center')
plt.text(x = 1, y = -0.25,s ='20-40',horizontalalignment= 'center')
plt.text(x = 2, y = -0.25,s ='40-60',horizontalalignment= 'center')
plt.text(x = 3, y = -0.25,s ='60-80',horizontalalignment= 'center')
plt.text(x = 4, y = -0.25,s ='80-100',horizontalalignment= 'center')
plt.text(x =-1.2, y = -0.09,s ='Actual Class:',horizontalalignment= 'center')
plt.text(x =-1.2, y = -0.25,s ='Age (years) :',horizontalalignment= 'center')
plt.yticks([])
plt.ylabel('Fraction of Predicted Class')
#plt.gcf().subplots_adjust(bottom=0.5)
#plt.gcf().subplots_adjust(left=0.15)
#fig.tight_layout( rect=[0, 0,2, 1])


#%% gender analysis
gender =[]
true_class = []
pred_class = []
for i in range(0,len(file_name)):
    for j in range(0,len(details_age)):
       # print(i,j)
        if file_name[i].split(" ")[1]== details_age[j][0]:
            #print('i am here')
            if details_age[j][4] !=  '':
                print(details_age[j])
                gender.append(details_age[j][4])
                true_class.append(np.argmax(true_cls[i]))
                pred_class.append(np.argmax(pred_prob[i]))
                break

gender_arr = np.asarray(gender)
class_arr = np.asarray(true_class)
pred_arr = np.asarray(pred_class)
#%%
def give_counts(gender_arr,class_arr, pred_arr,gender,true_class, pred_class):
    a = np.where(np.logical_and(class_arr==true_class, pred_arr==pred_class))
    b = np.where(gender_arr==gender)
    return np.intersect1d(a,b).size

color = ['#aed6dc', '#ff9a8d', '#4a536b']
#color = ['#e3b448', '#cbd18f', '#3a6b35']
#color = ['#316879', '#7fe7dc', '#f47a60']
xcolor1 = [  '#C1CD97','#E18D96','#78A2CC']
color2 = [  '#50874d','#A02C2D','#3065ac']
gender_range = ['M','F']
class_all = [0,1,2]
counts = np.zeros((3,3,2))
for k in class_all:
    for l in class_all:
        counter = 0
        for i in gender_range:
            counts[k,l,counter]=give_counts(gender_arr,class_arr, pred_arr,i,k,l)
            counter = counter+1

counts_norm = np.zeros((3,3,2))
for k in class_all:
    for l in class_all:
        for j in [0,1]: 
            counts_norm[k,l,j] =  counts[k,l,j]/sum(counts[k,:,j])
            
            
fig, ax = plt.subplots()
pos = [0,1.25]
w = [-.3,0,.3]
for k in class_all:
    bottom = [0,0]
    for l in class_all:
        counter = 0
        pos_shift = [z + w[k] for z in pos]
        plt.bar(pos_shift, counts_norm[k,l,:], bottom = bottom, color=color1[l], edgecolor=color2[l], width=.25)
        bottom = [a + b for a, b in zip(bottom, list(counts_norm[k,l,:]))]

plt.legend(['Nor','Pne','COV'],loc='lower center',bbox_to_anchor=(0.5, -0.35),ncol = 3,title="Predicted class")
#plt.xticks(ticks = pos,labels = ['Male','Female'],position = (-0.2,-0.2))
plt.xticks(ticks = [-0.3,0,0.3,.95,1.25,1.55],labels = ['Norm','Pne','COV','Norm','Pne','COV'],position = (-0.01,-0.01))
plt.text(x =-0.75, y = -0.08,s ='Actual Class:',horizontalalignment= 'center')
plt.text(x = 0, y = -0.15,s ='Male',horizontalalignment= 'center')
plt.text(x = 1.25, y = -0.15,s ='Female',horizontalalignment= 'center')
plt.text(x = -.75, y = -0.15,s ='Gender:',horizontalalignment= 'center')
plt.yticks([0,0.5,1])
plt.ylabel('Fraction of Predicted Class')
