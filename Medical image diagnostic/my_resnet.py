from my_metrics import average_class_specific_accuracy2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,LeakyReLU, ReLU

import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.optimizers import Adam

#%%
def get_resnet(seed):
    
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=None, input_shape=(224,224,3), pooling='avg', classes=1000)
    base_model.trainable = True
    
    x = base_model.output
    x = Dense(units = 1024, activation = 'relu', name = 'Dense_1')(x)
    x = Dense(units = 256, activation = 'relu', name = 'Dense_2')(x)
    x= Dense(units = 64, activation = 'relu', name = 'Dense_3')(x)
    x = Dense(units = 3, activation = 'softmax', name = 'Dense_4')(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy','AUC',average_class_specific_accuracy()]) # chnge average_class_specific_accuracy to average_class_specific_accuracy+str(no_class)
   # model.summary()
    return model



def denseMlpCreate(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    imIn=Input(shape=(1024,))

    x=Dense(256)(imIn)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(64)(x)
    x=LeakyReLU(alpha=0.1)(x)

    mlpFinal=Dense(2, activation='softmax')(x)

    mlp=Model(imIn, mlpFinal)
    mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.0002), metrics=['accuracy','AUC',average_class_specific_accuracy()]) # chnge average_class_specific_accuracy to average_class_specific_accuracy+str(no_class)

   # mlp.summary()
    
    return mlp

def denseMlpCreate_comb2(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    imIn=Input(shape=(2048,))
    
    x=Dense(256)(imIn)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(64)(x)
    x=LeakyReLU(alpha=0.1)(x)

    mlpFinal=Dense(7, activation='softmax')(x)

    mlp=Model(imIn, mlpFinal)
    mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.0002), metrics=['accuracy','AUC',average_class_specific_accuracy()]) # chnge average_class_specific_accuracy to average_class_specific_accuracy+str(no_class)

   # mlp.summary()
    
    return mlp



def denseMlpCreate_comb3(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    imIn=Input(shape=(3072,))
    
    x=Dense(1024)(imIn)
    x=LeakyReLU(alpha=0.1)(x)
    
    x=Dense(256)(x)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(64)(x)
    x=LeakyReLU(alpha=0.1)(x)

    mlpFinal=Dense(7, activation='softmax')(x)

    mlp=Model(imIn, mlpFinal)
    mlp.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.0002), metrics=['accuracy','AUC',average_class_specific_accuracy()]) # chnge average_class_specific_accuracy to average_class_specific_accuracy+str(no_class)

   # mlp.summary()
    
    return mlp

