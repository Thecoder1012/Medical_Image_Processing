import os
import random
import cv2
import numpy as np
#%%

def get_data(data_folder_path,classes, folders):
    random.seed(0)
    count = 0
    skipped_count = 0
    y = []
    folders_count = 0
    
    for folder in folders:
        print(folders_count)
        for labels in classes:
            files_list = os.listdir(os.path.join(data_folder_path,folder,labels))
            for files in files_list:
                img = cv2.imread(os.path.join(data_folder_path,folder,labels,files))
                if count == 0:
                    if img.shape == (50,50,3):
                        x=np.expand_dims(cv2.resize(img,(224,224)),axis = 0)
                        y.append(int(labels))
                    else:
                        skipped_count = skipped_count+1
                        print('skipped : ',labels,skipped_count)
                        
                else:
                    if img.shape == (50,50,3):
                        x = np.concatenate((x,np.expand_dims(cv2.resize(img,(224,224)),axis = 0)),axis = 0)
                        y.append(int(labels))
                    else:
                        skipped_count = skipped_count+1
                        print('skipped : ',labels,skipped_count )
                count = count +1
        folders_count = folders_count+1

    return x,y

#%%
random.seed(0)
path = './IDC' # add data path here
save_path = '' # add path to save the data
no_folders = 15
classes = ['0','1']
all_folders = os.listdir(path)
folders = random.sample(all_folders,no_folders)
data, labels = get_data(path,classes, folders)
np.savez(save_path+'/BHI/sample.npz', data, labels)
