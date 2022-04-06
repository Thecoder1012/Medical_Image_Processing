import csv
import random
import numpy as np
import cv2
import os


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    

def preprocess_image(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = np.asarray(cv2.resize(image, (224, 224)),dtype = np.float32)
   # image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

#%%
path = './kaggle_diabetic_ratinopathy/train' # add data path here
csv_path = './kaggle_diabetic_ratinopathy/trainLabels.csv' # add csv path here
save_path = '' # add path to save the data

classes = [0,1,2,3,4]
file_name = []
class_name = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count !=0:
            file_name.append(row[0])
            class_name.append(int(row[1]))
        line_count += 1
        
indices_list = []
for c in classes:
    indices = [i for i, x in enumerate(class_name) if x == c]
    if c == 0:
        indices = random.sample(indices,7000)
    indices_list.extend(indices)
    
count = 0
y = []
for i in indices_list:
    print(count)
    img = preprocess_image(os.path.join(path,file_name[i]+'.jpeg'))
    if count == 0:
        x=np.expand_dims(img,axis = 0)
        y.append(class_name[i])
    else:
        x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        y.append(class_name[i])
    count = count +1


np.savez(save_path+'/DR/sample.npz', x, y,file_name,file_name,indices_list)

    