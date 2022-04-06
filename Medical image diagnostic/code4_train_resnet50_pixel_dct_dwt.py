# import packages
from numpy import loadtxt
import numpy as np
import random
from my_metrics import average_class_specific_accuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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
from load_data import load_data, load_data_for_intermediate
import ast

#%%
data_type = 'dct' # enter data ype
data_name =  'DR' # enter dataset name

# %% set paths and parameters
seed = 0

save_path = '' # add the base path to save model and intermediate data
weight_path = save_path +'/'+data_type+'/'+data_name+'/'
intermedite_output_path = save_path+'/'+data_type+'_intermediate/'+data_name+'/'
batch_size = 32 
epochs = 300
input_size = 224

if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(intermedite_output_path):
    os.makedirs(intermedite_output_path)
#%%
def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
    
#%% 

trainS, labelTr, valS, labelVal, testS, labelTs = load_data(data_name,data_type)

#%%
y_train =np.concatenate((labelTr,labelVal)) 
weights = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)
classes = list(range(0,max(y_train)+1))
class_weights = {classes[i]: weights[i] for i in range(len(classes))}
del y_train
#%% define model
#seed = np.random.randint(0,1000)

reset_random_seeds(seed)
model = get_resnet(seed,len(classes))

#%% train
checkpoint = ModelCheckpoint(weight_path+'/checkpoint.h5', monitor='val_acsa', save_best_only=True, mode='max', verbose = 1)
earlystop = EarlyStopping(monitor='val_acsa', min_delta=0, patience=50, verbose=0, mode='max', baseline=None, restore_best_weights=False)
callbacks_list = [checkpoint,earlystop]
h = model.fit(x = trainS, y = to_categorical(labelTr), epochs=epochs, class_weight = class_weights, verbose=2,validation_data= (valS,to_categorical(labelVal)), shuffle=True,callbacks = callbacks_list)
model_json = model.to_json()
with open(weight_path+'/model.json', "w") as json_file:
    json_file.write(model_json)
with open(weight_path+'/model_history.json', 'w') as f:
    json.dump(str(h.history), f)

#%% evaluate
model.load_weights(weight_path+'/checkpoint.h5')
y_pred = model.predict(x = testS)
matrix = confusion_matrix(labelTs,np.argmax(y_pred,axis = 1))
matrix = matrix.astype('float')



# %% extract intermediate features 

model.load_weights(weight_path+'/checkpoint.h5')

     
intermediate_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("Dense_1").output)
intermediate_layer_model.summary()

data,label = load_data_for_intermediate(data_name,data_type)
#%%
intermediate_output = intermediate_layer_model.predict(data)
np.savez(intermedite_output_path+'/interm_data.npz',data = intermediate_output,label = label)

# for chest_xray and covid
# intermediate_output = intermediate_layer_model.predict(data[0])
# np.savez(intermedite_output_path+'/interm_data_train.npz',data = intermediate_output,label = label[0])

# intermediate_output = intermediate_layer_model.predict(data[1])
# np.savez(intermedite_output_path+'/interm_data_test.npz',data = intermediate_output,label = label[1])