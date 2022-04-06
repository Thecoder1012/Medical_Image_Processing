# This code computes DWT of the avaiable samples for each datasets 

# import packages
import cv2
import numpy as np
import pywt
#%%
path = '' # add data path here
#%% defintion

def mynorm(x):
    return (x-x.min())/(x.max()-x.min())

def compute_dwt(data):
    z = np.zeros_like(data)
    if data.dtype == 'uint8':
        z = np.asarray(z,dtype = np.uint8)
    else:
        z = np.asarray(z,dtype = np.float32)
    for i in range(0,data.shape[0]):
        print(i)
        img = data[i,:,:,:]
        h,w,ch= np.array(img.shape[:3])
        B = 112
        blocksV=int(h/B)
        blocksH=int(w/B)
        y = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            coeffs2 = pywt.dwt2(x, 'haar')
            # temp = [coeffs2[0], coeffs2[1][0], coeffs2[1][1], coeffs2[1][2]]
            temp = [mynorm(coeffs2[0]), mynorm(coeffs2[1][0]), mynorm(coeffs2[1][1]), mynorm(coeffs2[1][2])]
            count = 0
            for row in range(blocksV):
                    for col in range(blocksH):
                            y[row*B:(row+1)*B,col*B:(col+1)*B,channel]=temp[count]
                            count = count + 1
    if data.dtype == 'uint8':
        z[i,:] =  np.array(y*255,dtype = np.uint8)
    else:
        z[i,:] =  y
    return z

def compute_dwt_save(data_dir, text_file, save_path,B = 8):
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        img = np.asarray(cv2.imread(data_dir + file_name),dtype = np.float32)/255
        h,w,ch= np.array(img.shape[:3])
        B = 112
        blocksV=int(h/B)
        blocksH=int(w/B)
        y = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            coeffs2 = pywt.dwt2(x, 'haar')
            temp = [coeffs2[0], coeffs2[1][0], coeffs2[1][1], coeffs2[1][2]]
            count = 0
            for row in range(blocksV):
                    for col in range(blocksH):
                            y[row*B:(row+1)*B,col*B:(col+1)*B,channel]=temp[count]
                            count = count + 1
        np.save(save_path+'/'+file_name,y)
#%% this section is for chestxray1
txt_file_path = path+'/chestxray1/txtfiles/'
preprocessed_image_path = path+'/chestxray1/all_images/'
dwt_output_path = path+'/chestxray1/all_images_dwt/'

with open(txt_file_path+'merged'+'.txt', 'r') as fr:
     all_files = fr.readlines()
     
compute_dwt_save(preprocessed_image_path,all_files,dwt_output_path, B = 8)

#%% this section is for col_hist

temp = np.load(path+'/col_hist/sample.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/col_hist/sample_dwt.npz',data = dwt,label=label)
#%% this section is for isic18
temp = np.load(path+'/isic18/sample.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/isic18/sample_dwt.npz',data = dwt,label=label)
#%% This section is for chestxray2 (pneumonia)

temp = np.load(path+'/chestxray2/train.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/chestxray2/train_dwt.npz',data = dwt,label=label)

temp = np.load(path+'/chestxray2/test.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/chestxray2/test_dwt.npz',data = dwt,label=label)
#%% This section is for BHI
temp = np.load(path+'/BHI/sample.npz')
data = temp['arr_0']
label = temp['arr_1'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/BHI/sample_dwt.npz',data = dwt,label=label)

#%% This section is for CBISDDSM
temp = np.load(path+'/CBIS_DDSM/sample.npz')
data = temp['data']
label = temp['label'] 
del temp

dwt = compute_dwt(data)
np.savez(path+'/CBIS_DDSM/sample_dwt.npz',data = dwt, label = label)
#%% This section is for DR
temp = np.load(path+'/DR/sample.npz')
data = temp['arr_0']
label = np.asarray(temp['arr_1'])
dwt = compute_dwt(data)
del data
np.savez(path+'/DR/sample_dwt.npz',data = dwt, label = label)
