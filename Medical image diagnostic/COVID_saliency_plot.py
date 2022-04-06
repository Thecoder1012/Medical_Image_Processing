# import packages
from numpy import loadtxt
import numpy as np
import random
from my_metrics import average_class_specific_accuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data import *
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
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
import matplotlib.pyplot as plt
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
# %% set paths and parameters
txt_file_path = './chestxray1/txtfiles/'
preprocessed_image_path = './chestxray1/all_images/'
preprocessed_dct_path = './chestxray1/all_images_dct/'
preprocessed_dwt_path = './chestxray1/all_images_dwt/'
#%%
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
    weights = []
    mapping={
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        x.append(cv2.imread(data_dir + file_name))
        y.append(line_content[i].split(" ")[2].rstrip())
    y =list(map(mapping.get, y))
    for j in range(0,len(list(set(y)))):
        weights.append(y.count(list(set(y))[j])) 
    weights = [element / sum(weights) for element in weights]
    return np.asarray(x,dtype = np.float32)/255, to_categorical(np.asarray(y,dtype = np.float32)), weights

def get_data_label_dcwt(data_dir,text_file): 
    x = []
    y = []
    weights = []
    mapping={
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1].rstrip()
        x.append(np.load(data_dir + file_name+'.npy'))
        y.append(line_content[i].split(" ")[2].rstrip())
    y =list(map(mapping.get, y))
    for j in range(0,len(list(set(y)))):
        weights.append(y.count(list(set(y))[j])) 
    weights = [element / sum(weights) for element in weights]
    return np.asarray(x,dtype = np.float32), to_categorical(np.asarray(y,dtype = np.float32)), weights

def model_modifier(m):
    m.layers[-1].activation = None#tf.keras.activations.linear
    return m

def loss0(output):
   # print(output.shape)
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][0])

def loss1(output):
   # print(output.shape)
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][1])

def loss2(output):
   # print(output.shape)
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][2])



def get_data_label_int(data_dir,text_file): 
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

#%% saliency for combined 
weight_path1= './pixel/chestxray1/'
weight_path1= './dct/chestxray1/'
weight_path1= './dwt/chestxray1/'
mlp_weight = './pixel_dct_dwt/chestxray1/mlp_weight/'

seed = 0#np.random.randint(0,1000)
model1 = get_resnet(seed)
model2 = get_resnet(seed)
model3 = get_resnet(seed)
mlp = denseMlpCreate_comb3(seed)
mlp.load_weights(mlp_weight+'checkpoint.h5')
model1.load_weights(weight_path1+'checkpoint.h5')
model2.load_weights(weight_path2+'checkpoint.h5')
model3.load_weights(weight_path3+'checkpoint.h5')

for i, layer in enumerate(model1.layers):
    layer._name = 'layer_1_' + str(i)
for i, layer in enumerate(model2.layers):
    layer._name = 'layer_2_' + str(i)
for i, layer in enumerate(model3.layers):
    layer._name = 'layer_3_' + str(i)
    
x1 = model1.layers[176].output
x2 = model2.layers[176].output
x3 = model3.layers[176].output

x  = concatenate([x1,x2,x3])
for i, layer in enumerate(mlp.layers):
    x = mlp.layers[i](x)
    
model=Model([model1.layers[0].input,model2.layers[0].input,model3.layers[0].input], x)
#model=Model(model2.layers[0].input, model2.layers[176].output)


#k = 2
data_type = 'dct'
class_type = 'covid'
if class_type == 'normal':
    class_id = 0
    n1 = range(0, 100)
elif class_type == 'pneumonia':
    class_id = 1
    n1 = range(100, 200)
elif class_type == 'covid':
    class_id = 2
    n1 = range(200, 300) 
if data_type == 'raw':
    cam_id = 0
elif data_type == 'dct':
    cam_id = 1
elif data_type == 'dwt':
    cam_id = 2

save_path = './chestxray1/saliency/plot/combined/'+data_type+'/'+class_type+'/'
    
with open('./chestxray1/txtfiles/test.txt', 'r') as fr:
      test_files = fr.readlines()
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for j in n1:
    test_file = [test_files[j]]
    x, y, _ = get_data_label(preprocessed_image_path,test_file)
    file_name = test_file[0].split(" ")[1]
    images = x
    images = images.astype('float32')
    X1 = np.asarray(images)
    img = images[0,:,:,:]
    norm_im1 = (img -img.min())/(img.max()-img.min())
    
    x, y, _ = get_data_label_dcwt(preprocessed_dct_path,test_file)
    file_name = test_file[0].split(" ")[1]
    images = x
    images = images.astype('float32')
    X2 = np.asarray(images)
    img = images[0,:,:,:]
    norm_im2 = (img -img.min())/(img.max()-img.min())
    
    x, y, _ = get_data_label_dcwt(preprocessed_dwt_path,test_file)
    file_name = test_file[0].split(" ")[1]
    images = x
    images = images.astype('float32')
    X3 = np.asarray(images)
    im, _, _ = get_data_label_dcwt(preprocessed_dwt_norm_path,test_file)
    img = np.zeros((224,224,3))
    img[:,:,0] = im
    img[:,:,1] = im
    img[:,:,2] = im
    norm_im3 = (img -img.min())/(img.max()-img.min())
    
    y_pred = model.predict([X1,X2,X3])
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)
    if np.argmax(y_pred)==class_id: 
        if class_id==0:
            cam = saliency(loss0,[X1,X2,X3],smooth_samples=0, smooth_noise=0)
        elif class_id==1:
            cam = saliency(loss1,[X1,X2,X3],smooth_samples=0, smooth_noise=0)
        elif class_id==2:
            cam = saliency(loss2,[X1,X2,X3],smooth_samples=0, smooth_noise=0)
        #cam = gradcam(loss0,[X1,X2,X3],penultimate_layer=-1)
        #cam = scorecam(loss0,[X1,X2,X3],penultimate_layer=-1, max_N=10)
        for i in [0,1,2]:
            cam[i] =2*(cam[i]-cam[i].min())/(cam[i].max()-cam[i].min())
        #     for ii in range(0,1):
        #         for iii in range(0,224):
        #             for iiii in range(0,224):
        #                 cam[i][ii,iii,iiii] = np.maximum(cam[i][ii,iii,iiii],0)
            #cam[i] = cam[i]/cam[i].max()
    
        #       #mu = np.mean(cam[i])
        #       #std = np.std(cam[i])
        #       #cam[i] = (cam[i]-mu)/std
        #       #cam[i] = normalize(cam[i])
        #     print(cam[i].min(),cam[i].max())
        
        f, ax = plt.subplots(1,1)
        heatmap = np.uint8(cm.jet((cam[cam_id][0]))[..., :3] * 255)
        
        heatmap = cv2.fastNlMeansDenoisingColored(heatmap,None,30,3,7,21)
        heatmap = heatmap.astype(int)    
        ax.imshow(norm_im2)
        ax.imshow(heatmap, cmap='jet', alpha=0.3) # overlay
        plt.tight_layout()
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

        f.savefig(save_path+file_name)
       # plt.show()
#%% define separate types of features 
weight_path1= './pixel/chestxray1/'
weight_path1= './dct/chestxray1/'
weight_path1= './dwt/chestxray1/'

data_type = 'dct' # dwt or dct or pixel
class_type = 'covid'
if class_type == 'normal':
    class_id = 0
    n1 = range(0, 100)
elif class_type == 'pneumonia':
    class_id = 1
    n1 = range(100, 200)
elif class_type == 'covid':
    class_id = 2
    n1 = range(200, 300) 
    
if data_type == 'raw':
    cam_id = 0
    mlp_weight = '/pixel/chestxray1/mlp_weight/'
    weight_path = weight_path1
elif data_type == 'dct':
    cam_id = 1
    mlp_weight = '/dct/chestxray1/mlp_weight/'
    weight_path = weight_path2
elif data_type == 'dwt':
    cam_id = 2
    mlp_weight = '/dwt/chestxray1/mlp_weight/'
    weight_path = weight_path3

save_path = './chestxray1/saliency/plot/single/'+data_type+'/'+class_type+'/'
    
seed = 0#np.random.randint(0,1000)
model1 = get_resnet(seed)
mlp = denseMlpCreate(seed)
mlp.load_weights(mlp_weight+'checkpoint.h5')
model1.load_weights(weight_path+'checkpoint.h5')

x = model1.layers[176].output
#x2 = model2.layers[176].output
#x3 = model3.layers[176].output

#x  = concatenate([x1,x2,x3])
for i, layer in enumerate(mlp.layers):
    x = mlp.layers[i](x)
    
model=Model([model1.layers[0].input], x)
        

with open('./chestxray1/txtfiles/test.txt', 'r') as fr:
      test_files = fr.readlines()
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for j in n1:
    test_file = [test_files[j]]
    if data_type == 'raw': 
        x, y, _ = get_data_label(preprocessed_image_path,test_file)
        file_name = test_file[0].split(" ")[1]
        images = x
        images = images.astype('float32')
        X = np.asarray(images)
        img = images[0,:,:,:]
        norm_im = (img -img.min())/(img.max()-img.min())
        
    elif data_type == 'dct':
        x, y, _ = get_data_label_dcwt(preprocessed_dct_path,test_file)
        file_name = test_file[0].split(" ")[1]
        images = x
        images = images.astype('float32')
        X = np.asarray(images)
        img = images[0,:,:,:]
        norm_im = (img -img.min())/(img.max()-img.min())

    elif data_type == 'dwt':
        x, y, _ = get_data_label_dcwt(preprocessed_dwt_path,test_file)
        file_name = test_file[0].split(" ")[1]
        images = x
        images = images.astype('float32')
        X = np.asarray(images)
        im, _, _ = get_data_label_dcwt(preprocessed_dwt_norm_path,test_file)
        img = np.zeros((224,224,3))
        img[:,:,0] = im
        img[:,:,1] = im
        img[:,:,2] = im
        norm_im = (img -img.min())/(img.max()-img.min())
        
    y_pred = model.predict(X)
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)
    if np.argmax(y_pred)==class_id: 
        if class_id==0:
            cam = saliency(loss0,X,smooth_samples=50, smooth_noise=0.002)
        elif class_id==1:
            cam = saliency(loss1,X,smooth_samples=50, smooth_noise=0.002)
        elif class_id==2:
            cam = saliency(loss2,X,smooth_samples=50, smooth_noise=0.002)
        #cam = gradcam(loss0,[X1,X2,X3],penultimate_layer=-1)
        #cam = scorecam(loss0,[X1,X2,X3],penultimate_layer=-1, max_N=10)
        for i in [0]:
            cam[i] =(cam[i]-cam[i].min())/(cam[i].max()-cam[i].min())
        #     for ii in range(0,1):
        #         for iii in range(0,224):
        #             for iiii in range(0,224):
        #                 cam[i][ii,iii,iiii] = np.maximum(cam[i][ii,iii,iiii],0)
            #cam[i] = cam[i]/cam[i].max()
    
        #       #mu = np.mean(cam[i])
        #       #std = np.std(cam[i])
        #       #cam[i] = (cam[i]-mu)/std
        #       #cam[i] = normalize(cam[i])
        #     print(cam[i].min(),cam[i].max())
        
        f, ax = plt.subplots(1,1)
        heatmap = cm.jet((cam[0]))[..., :3] * 255
        
        #heatmap = cv2.fastNlMeansDenoisingColored(heatmap,None,30,3,7,21)
        heatmap = heatmap.astype(int)    
        ax.imshow(norm_im)
        ax.imshow(heatmap, cmap='jet', alpha=0.3) # overlay
        plt.tight_layout()
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_xlabel('(a)')
        # heatmap = np.uint8(cm.jet((cam[1][0]))[..., :3] * 255)
        # print(heatmap.min(), heatmap.max())
        # ax[1].imshow(norm_im2)
        # ax[1].imshow(heatmap, cmap='jet', alpha=0.3) # overlay
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # heatmap = np.uint8(cm.jet((cam[2][0]))[..., :3] * 255)
        # ax[2].imshow(norm_im3)
        # ax[2].imshow(heatmap, cmap='jet', alpha=0.3) # overlay
        # ax[2].set_xticks([])
        # ax[2].set_yticks([])
        f.savefig(save_path+file_name)
       # plt.show()