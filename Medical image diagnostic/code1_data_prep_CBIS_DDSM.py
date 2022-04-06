import tensorflow as tf
import cv2
import numpy as np
#%%
images=[]
labels=[]
feature_dictionary = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_normal': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
    }

def _parse_function(example, feature_dictionary=feature_dictionary):
    parsed_example = tf.io.parse_example(example, feature_dictionary)
    return parsed_example

def read_data(filename):
    full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # full_dataset = full_dataset.shuffle(buffer_size=31000)
    full_dataset = full_dataset.cache()
    print("Size of Training Dataset: ", len(list(full_dataset)))
    
    feature_dictionary = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_normal': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
    }   

    full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image_features in full_dataset:
        image = image_features['image'].numpy()
        image = tf.io.decode_raw(image_features['image'], tf.uint8)
        image = tf.reshape(image, [299, 299])        
        image=image.numpy()
        image=cv2.resize(image,(224,224))
        image=cv2.merge([image,image,image])        
        #plt.imshow(image)
        images.append(image)
        labels.append(image_features['label_normal'].numpy())

filenames=['D:/Simpi/data/IDC/training10_0/training10_0.tfrecords',
          # 'D:/Simpi/data/IDC/training10_1/training10_1.tfrecords',
          # 'D:/Simpi/data/IDC/training10_2/training10_2.tfrecords',
          # 'D:/Simpi/data/IDC/training10_3/training10_3.tfrecords',
          # 'D:/Simpi/data/IDC/training10_4/training10_4.tfrecords']
          ]
    
for file in filenames:
    read_data(file)

#%%
data = np.stack(images, axis=0 )
label = np.asarray(labels)
save_path = '' # add path to save the data
np.savez(save_path+'/CBIS_DDSM/sample.npz',data = data, label = label)