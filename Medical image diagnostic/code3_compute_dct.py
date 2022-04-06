# This code computes DCT of the avaiable samples for each datasets 

# import packages
import cv2
import numpy as np
#%%
path = '' # add data path here
#%%
def mynorm(x):
    return (x-x.min())/(x.max()-x.min())

def compute_dct(data, B = 8):
    y = np.zeros_like(data)
    if data.dtype == 'uint8':
        y = np.asarray(y,dtype = np.uint8)
    else:
        y = np.asarray(y,dtype = np.float32)
    for i in range(0,data.shape[0]):
        print(i)
        img = data[i,:,:,:]
        h,w,ch= np.array(img.shape[:3])
        blocksV=int(h/B)
        blocksH=int(w/B)
        Trans = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            vis0 = np.zeros((h,w), np.float32)
            vis0[:h, :w] = x
            for row in range(blocksV):
                    for col in range(blocksH):
                            currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                            Trans[row*B:(row+1)*B,col*B:(col+1)*B,channel]=currentblock
    if data.dtype == 'uint8':
        y[i,:] = np.array(mynorm(Trans)*255,dtype = np.uint8)
    else:
        y[i,:] = mynorm(Trans)
    return y
    
def compute_dct_save(data_dir, text_file, save_path,B = 8):
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        img = np.asarray(cv2.imread(data_dir + file_name),dtype = np.float32)/255
        h,w,ch= np.array(img.shape[:3])
        blocksV=int(h/B)
        blocksH=int(w/B)
        Trans = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            vis0 = np.zeros((h,w), np.float32)
            vis0[:h, :w] = x
            for row in range(blocksV):
                    for col in range(blocksH):
                            currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                            Trans[row*B:(row+1)*B,col*B:(col+1)*B,channel]=currentblock
        np.save(save_path+'/'+file_name,Trans)
#%% this section is for chestxray1
txt_file_path = path+'/chestxray1/txtfiles/'
preprocessed_image_path = path+'/chestxray1/all_images/'
dct_output_path = path+'/chestxray1/all_images_dct/'

with open(txt_file_path+'merged'+'.txt', 'r') as fr:
     all_files = fr.readlines()
     
compute_dct_save(preprocessed_image_path,all_files,dct_output_path, B = 8)

#%%  this section is for col_hist dataset
temp = np.load(path+'/col_hist/sample.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/col_hist/sample_dct.npz',data = dct,label=label)
#%% this section is for ISIC18 datasets
temp = np.load(path+'/isic18/sample.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/isic18/sample_dct.npz',data = dct,label=label)
#%% This section is for chestxray2 (pneumonia)
temp = np.load(path+'/chestxray2/train.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/chestxray2/train_dct.npz',data = dct,label=label)

temp = np.load(path+'/chestxray2/test.npz')
data = temp['arr_0']
#label = np.asarray(temp['arr_1'])
label = temp['arr_1'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/chestxray2/test_dct.npz',data = dct,label=label)


#%% this section is for BHI
temp = np.load(path+'/BHI/sample.npz')
data = temp['arr_0']
label = temp['arr_1'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/BHI/sample_dct.npz',data = dct,label=label)
#%% This section is for CBISDDSM
temp = np.load(path+'/CBIS_DDSM/sample.npz')
data = temp['data']
label = temp['label'] 
del temp

dct = compute_dct(data, B = 8)
np.savez(path+'/CBIS_DDSM/sample_dct.npz',data = dct, label = label)
#%% This section is for DR
temp = np.load(path+'/DR/sample.npz')
data = temp['arr_0']
label = np.asarray(temp['arr_1'])

del temp
dct = compute_dct(data, B = 8)
np.savez(path+'/DR/sample_dct.npz',data = dct, label = label)