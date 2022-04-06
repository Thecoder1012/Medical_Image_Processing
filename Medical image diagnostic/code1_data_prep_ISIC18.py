import os
import numpy as np
import csv
import cv2
#%%

path = './ISIC2018_Task3_Training_Input' # add data path here
csv_path = './ISIC_grounfthruth.csv' # add csv path here
save_path = '' # add path to save the data

classes = [0,1,2,3,4]
file_name = []
y = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(line_count)
        if line_count !=0:
            file_name.append(row[0])
            img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,row[0]+'.jpg')),(224,224))/255, dtype =np.float32)
            y.append(np.argmax((row[1::])))
            if line_count ==1:
                x=np.expand_dims(img,axis = 0)
            else:
                x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        line_count += 1
        
np.savez(save_path+'/isic18/sample.npz', x, y,file_name)

