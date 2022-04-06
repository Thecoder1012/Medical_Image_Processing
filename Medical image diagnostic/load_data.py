import numpy as np
from sklearn.model_selection import train_test_split
import cv2
def load_data(data_name,data_type ):
    path = ''# add base path to the data here
    if data_name == 'col_hist':
        if data_type == 'pixel':
            temp = np.load(path+'/col_hist/sample.npz')
            x, testS, y, labelTs = train_test_split(temp['arr_0'], temp['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/col_hist/sample_dct.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/col_hist/sample_dwt.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
            
    elif data_name == 'ISIC18':
        if data_type == 'pixel':
            temp = np.load(path+'/isic18/sample.npz')
            x, testS, y, labelTs = train_test_split(temp['arr_0'], temp['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/isic18/sample_dct.npz')
            x, testS, y, labelTs = train_test_split(temp['arr_0'], temp['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/isic18/sample_dwt.npz')
            x, testS, y, labelTs = train_test_split(temp['arr_0'], temp['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
            
    elif data_name == 'chestxray2':
        if data_type == 'pixel':
            temp1 = np.load(path+'/chestxray2/train.npz')
            temp2 = np.load(path+'/chestxray2/test.npz')
            trainS, valS, labelTr, labelVal = train_test_split(temp1['arr_0'], temp1['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            testS, labelTs = temp2['arr_0'],temp2['arr_1']
            
            del temp1,temp2
            
        elif data_type == 'dct':
            temp1 = np.load(path+'/chestxray2/train_dct.npz')
            temp2 = np.load(path+'/chestxray2/test_dct.npz')
            trainS, valS, labelTr, labelVal = train_test_split(temp1['data'], temp1['label'], test_size=0.2, random_state=0,shuffle=True)
            testS, labelTs = temp2['data'],temp2['label']
            
            del temp1,temp2
        elif data_type == 'dwt':
            temp1 = np.load(path+'/chestxray2/train_dwt.npz')
            temp2 = np.load(path+'/chestxray2/test_dwt.npz')
            trainS, valS, labelTr, labelVal = train_test_split(temp1['data'], temp1['label'], test_size=0.2, random_state=0,shuffle=True)
            testS, labelTs = temp2['data'],temp2['label']
            
            del temp1,temp2
            
    elif data_name == 'BHI':
        if data_type == 'pixel':
            temp = np.load(path+'/BHI/sample.npz')
            x, testS, y, labelTs = train_test_split(temp['arr_0'], temp['arr_1'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
            
        elif data_type == 'dct':
            temp = np.load(path+'/BHI/sample_dct.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/BHI/sample_dwt.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
            
    elif data_name == 'CBIS_DDSM':
        if data_type == 'pixel':
            temp = np.load(path+'/CBIS_DDSM/sample.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/CBIS_DDSM/sample_dct.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/CBIS_DDSM/sample_dwt.npz')
            x, testS, y, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp

    elif data_name == 'DR':
        if data_type == 'pixel':
            temp = np.load(path+'/DR/sample.npz')
            label = temp['arr_1']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1            
            x, testS, y, labelTs = train_test_split(temp['arr_0'], label, test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/DR/sample_dct.npz')
            label = temp['label']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1   
            x, testS, y, labelTs = train_test_split(temp['data'], label, test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/DR/sample_dwt.npz')
            label = temp['label']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1   
            x, testS, y, labelTs = train_test_split(temp['data'], label, test_size=0.2, random_state=0,shuffle=True)
            trainS, valS, labelTr, labelVal = train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)
            
            del temp
            
    elif data_name == 'chestxray1':

        train_txt =  path+'/chestxray1/txtfiles/train.txt'
        val_txt =  path+'/chestxray1/txtfiles/val.txt'
        test_txt =  path+'/chestxray1/txtfiles/test.txt'
        with open(test_txt, 'r') as fr:
             test_files = fr.readlines()
        with open(val_txt, 'r') as fr:
             val_files = fr.readlines()
        with open(train_txt, 'r') as fr:
             train_files = fr.readlines()
        
        train_files  = train_files 

        if data_type == 'pixel':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(cv2.imread(data_dir + file_name))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32)/255, y
    
            preprocessed_image_path = path+'/chestxray1/all_images/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            valS, labelVal = get_data_label(preprocessed_image_path,val_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)
            
        elif data_type == 'dct':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(np.load(data_dir + file_name+'.npy'))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32), y
    

            preprocessed_image_path = path+'/chestxray1/all_images_dct/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            valS, labelVal = get_data_label(preprocessed_image_path,val_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)
            
        elif data_type == 'dwt':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(np.load(data_dir + file_name+'.npy'))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32), y
    
            preprocessed_image_path = path+'/chestxray1/all_images_dwt/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            valS, labelVal = get_data_label(preprocessed_image_path,val_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)


    return trainS,labelTr, valS, labelVal, testS, labelTs

def load_intermediate_data(data_name,data_type ):
    path = ''# add base path to the data here
    if data_name == 'col_hist':
        if data_type == 'pixel':
            temp = np.load(path+'/pixel_intermediate/col_hist/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/dct_intermediate/col_hist/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/dwt_intermediate/col_hist/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
    elif data_name == 'ISIC18':
        if data_type == 'pixel':
            temp = np.load(path+'/pixel_intermediate/isic18/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/dct_intermediate/isic18/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/dwt_intermediate/isic18/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
            
    elif data_name == 'chestxray2':
        if data_type == 'pixel':
            temp1 = np.load(path+'/pixel_intermediate/chestxray2/interm_data_train.npz')
            temp2 = np.load(path+'/pixel_intermediate/chestxray2/interm_data_test.npz')

            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2

        elif data_type == 'dct':
            temp1 = np.load(path+'/dct_intermediate/chestxray2/interm_data_train.npz')
            temp2 = np.load(path+'/dct_intermediate/chestxray2/interm_data_test.npz')
            
            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2
            
        elif data_type == 'dwt':
            temp1 = np.load(path+'/dwt_intermediate/chestxray2/interm_data_train.npz')
            temp2 = np.load(path+'/dwt_intermediate/chestxray2/interm_data_test.npz')
            
            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2
            
    elif data_name == 'BHI':
        if data_type == 'pixel':
            temp = np.load(path+'/pixel_intermediate/BHI/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/dct_intermediate/BHI/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/dwt_intermediate/BHI/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        
    elif data_name == 'CBIS_DDSM':
        if data_type == 'pixel':
            temp = np.load(path+'/pixel_intermediate/CBIS_DDSM/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/dct_intermediate/CBIS_DDSM/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/dwt_intermediate/CBIS_DDSM/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
            

    elif data_name == 'DR':
        if data_type == 'pixel':
            temp = np.load(path+'/pixel_intermediate/DR/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/dct_intermediate/DR/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/dwt_intermediate/DR/interm_data.npz')
            trainS, testS, labelTr, labelTs = train_test_split(temp['data'], temp['label'], test_size=0.2, random_state=0,shuffle=True)

            del temp
            
                
    elif data_name == 'chestxray1':
        if data_type == 'pixel':
            temp1 = np.load(path+'/pixel_intermediate/chestxray1/interm_data_train.npz')
            temp2 = np.load(path+'/pixel_intermediate/chestxray1/interm_data_test.npz')
            
            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2

        elif data_type == 'dct':
            temp1 = np.load(path+'/dct_intermediate/chestxray1/interm_data_train.npz')
            temp2 = np.load(path+'/dct_intermediate/chestxray1/interm_data_test.npz')
            
            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2
            
        elif data_type == 'dwt':
            temp1 = np.load(path+'/dwt_intermediate/chestxray1/interm_data_train.npz')
            temp2 = np.load(path+'/dwt_intermediate/chestxray1/interm_data_test.npz')
            
            trainS = temp1['data']
            testS = temp2['data']
            labelTr = temp1['label']
            labelTs = temp2['label']
            
            del temp1,temp2

    return trainS, labelTr, testS, labelTs



def load_data_for_intermediate(data_name,data_type ):
    path = ''# add base path to the data here
    if data_name == 'col_hist':
        if data_type == 'pixel':
            temp = np.load(path+'/col_hist/sample.npz')
            data = temp['arr_0']
            label = temp['arr_1']
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/col_hist/sample_dct.npz')
            data = temp['data']
            label = temp['label']
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/col_hist/sample_dwt.npz')
            data = temp['data']
            label = temp['label']
            del temp
            
    elif data_name == 'ISIC18':
        if data_type == 'pixel':
            temp = np.load(path+'/isic18/sample.npz')
            data = temp['arr_0']
            label = temp['arr_1']
            del temp
        elif data_type == 'dct':
            temp = np.load(path+'/isic18/sample_dct.npz')
            data = temp['data']
            label = temp['label']
            del temp
        elif data_type == 'dwt':
            temp = np.load(path+'/isic18/sample_dwt.npz')
            data = temp['data']
            label = temp['label']
            del temp
            
    elif data_name == 'chestxray2':
        if data_type == 'pixel':
            temp1 = np.load(path+'/chestxray2/train.npz')
            temp2 = np.load(path+'/chestxray2/test.npz')
            data = [temp1['arr_0'],temp2['arr_0']]
            label = [temp1['arr_1'],temp2['arr_1']]
            del temp1,temp2
            
        elif data_type == 'dct':
            temp1 = np.load(path+'/chestxray2/train_dct.npz')
            temp2 = np.load(path+'/chestxray2/test_dct.npz')
            data = [temp1['data'],temp2['data']]
            label = [temp1['label'],temp2['label']]
            
            del temp1,temp2
        elif data_type == 'dwt':
            temp1 = np.load(path+'/chestxray2/train_dwt.npz')
            temp2 = np.load(path+'/chestxray2/test_dwt.npz')
            data = [temp1['data'],temp2['data']]
            label = [temp1['label'],temp2['label']]
            
            del temp1,temp2
        
    elif data_name == 'BHI':
        if data_type == 'pixel':
            temp = np.load(path+'/BHI/sample.npz')
            data = temp['arr_0']
            label = temp['arr_1']
            del temp
        if data_type == 'dct':
            temp = np.load(path+'/BHI/sample_dct.npz')
            data = temp['data']
            label = temp['label']
            del temp
        if data_type == 'dwt':
            temp = np.load(path+'/BHI/sample_dwt.npz')
            data = temp['data']
            label = temp['label']
            del temp
            
    elif data_name == 'CBIS_DDSM':
        if data_type == 'pixel':
            temp = np.load(path+'/CBIS_DDSM/sample.npz')
            data = temp['data']
            label = temp['label']
            del temp
        if data_type == 'dct':
            temp = np.load(path+'/CBIS_DDSM/sample_dct.npz')
            data = temp['data']
            label = temp['label']
            del temp
        if data_type == 'dwt':
            temp = np.load(path+'/CBIS_DDSM/sample_dwt.npz')
            data = temp['data']
            label = temp['label']
            del temp
            
    elif data_name == 'DR':
        if data_type == 'pixel':
            temp = np.load(path+'/DR/sample.npz')
            data = temp['arr_0']
            label = temp['arr_1']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1
            del temp
        if data_type == 'dct':
            temp = np.load(path+'/DR/sample_dct.npz')
            data = temp['data']
            label = temp['label']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1
            del temp
        if data_type == 'dwt':
            temp = np.load(path+'/DR/sample_dwt.npz')
            data = temp['data']
            label = temp['label']
            for n, i in enumerate(label):
                if i > 0:
                    label[n] = 1
            del temp
            
    elif data_name == 'chestxray1':

        train_txt =  path+'/chestxray1/txtfiles/train.txt'
        val_txt =  path+'/chestxray1/txtfiles/val.txt'
        test_txt =  path+'/chestxray1/txtfiles/test.txt'
        with open(test_txt, 'r') as fr:
             test_files = fr.readlines()
        with open(val_txt, 'r') as fr:
             val_files = fr.readlines()
        with open(train_txt, 'r') as fr:
             train_files = fr.readlines()
        
        train_files  = train_files + val_files

        if data_type == 'pixel':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(cv2.imread(data_dir + file_name))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32)/255, y
    
            preprocessed_image_path = path+'/chestxray1/all_images/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)
            data = [trainS,testS]
            label = [labelTr,labelTs]
            
        elif data_type == 'dct':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(np.load(data_dir + file_name+'.npy'))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32), y
    

            preprocessed_image_path = path+'/chestxray1/all_images_dct/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)
            data = [trainS,testS]
            label = [labelTr,labelTs]
            
        elif data_type == 'dwt':
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
                    file_name = line_content[i].split(" ")[1].rstrip()
                    x.append(np.load(data_dir + file_name+'.npy'))
                    y.append(line_content[i].split(" ")[2].rstrip())
                y =list(map(mapping.get, y))
                for j in range(0,len(list(set(y)))):
                    weights.append(y.count(list(set(y))[j])) 
                weights = [element / sum(weights) for element in weights]
                return np.asarray(x,dtype = np.float32), y
    
            preprocessed_image_path = path+'/chestxray1/all_images_dwt/'
            trainS, labelTr = get_data_label(preprocessed_image_path,train_files)
            testS, labelTs= get_data_label(preprocessed_image_path,test_files)
            data = [trainS,testS]
            label = [labelTr,labelTs]
    return data,label
