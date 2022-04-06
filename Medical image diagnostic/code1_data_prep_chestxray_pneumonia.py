import os
import numpy as np
import csv
import cv2
#%%

classes = [0,1]
y = []
count = 0
class_count = 0
path = './chestxray2/train' # add data path here
save_path = '' # add path to save the data
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/chestxray2/train.npz', x, y,file_name)


classes = [0,1]
y = []
count = 0
class_count = 0
path = './chestxray2/test' # add data path here
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/chestxray2/test.npz', x, y,file_name)
