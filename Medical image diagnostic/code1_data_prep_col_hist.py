import numpy as np
from PIL import Image 
import os

#%%
path = './Kather_texture_2016_image_tiles_5000' # add data path here
save_path = '' # add path to save the data

x = []
y = []
dirs = os.listdir(path)
count = 0
class_name = 0
for folder in dirs:
    print(count)
    file_list = os.listdir(os.path.join(path,folder))
    for file in file_list:
        img = Image.open(os.path.join(path,folder,file))
        img = np.asarray(img.resize((224,224)),dtype = np.float32)/255
        if count == 0:
            x=np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
    class_name = class_name+1


np.savez(save_path+'/col_hist/sample.npz', x, y)
