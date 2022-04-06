
#%% cohen metadat load
path = './covid-chestxray-dataset/metadata.csv'

from csv import reader
# read csv file as a list of lists
with open(path, 'r', encoding="utf-8") as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    cohen = list(csv_reader)
#%% fig1
path = './Figure1-COVID-chestxray-dataset/metadata.csv'

from csv import reader
# read csv file as a list of lists
with open(path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    fig1 = list(csv_reader)
#%% actmed
path = './Actualmed-COVID-chestxray-dataset/metadata.csv'

from csv import reader
# read csv file as a list of lists
with open(path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    actmed = list(csv_reader)   
#%% sirm
path = './COVID-19-Radiography-Database/COVID-19.metadata_details.csv'

from csv import reader
# read csv file as a list of lists
with open(path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    sirm = list(csv_reader)   

#%% rsna
path = './Data_Entry_2017_v2020.csv'

from csv import reader
# read csv file as a list of lists
with open(path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    rsna = list(csv_reader)   
    
#%%
txt_file_path = './chestxray1/txtfiles/merged.txt'
with open(txt_file_path, 'r') as fr:
    all_files = fr.readlines()
#%%
a = []
for i in range(0,len(all_files)):
    a.append(all_files[i].split(' ')[0])
b = list(set(a))

index = []
for i in b:
    index.append(b.index(i))

all_files1 = [all_files[i] for i in index]
all_files = all_files1 
#%%
details_cohen = []
for i in range(0,len(all_files)):
    file_name = all_files[i].split(" ")[1]
    source = all_files[i].split(" ")[3].rstrip()
    class_name =  all_files[i].split(" ")[2]
    for j in range(1,len(cohen)):
        f_name = cohen[j][22]
        if file_name == f_name:
            age = cohen[j][3]
            sex = cohen[j][2]
            print(f_name)
            details_cohen.append([file_name, source, class_name, age,sex])
            break
#%%
details_fig1 = []
for i in range(0,len(all_files)):
    file_name = all_files[i].split(" ")[1][:-4]
    source = all_files[i].split(" ")[3].rstrip()
    class_name =  all_files[i].split(" ")[2]
    for j in range(1,len(fig1)):
        f_name = fig1[j][0]
        if file_name == f_name:
            age = fig1[j][3]
            sex = fig1[j][2]
            print(f_name)
            details_fig1.append([file_name, source, class_name, age,sex])
            break
#%%
details_actmed = []
for i in range(0,len(all_files)):
    file_name = all_files[i].split(" ")[0]
    source = all_files[i].split(" ")[3].rstrip()
    class_name =  all_files[i].split(" ")[2]
    for j in range(1,len(actmed)):
        f_name = actmed[j][0]
        if file_name == f_name:
            age = actmed[j][3]
            sex = actmed[j][2]
            print(f_name)
            details_actmed.append([all_files[i].split(" ")[1], source, class_name, age,sex])
            break
#%%
details_sirm = []
for i in range(0,len(all_files)):
    file_name = all_files[i].split(" ")[0]
    source = all_files[i].split(" ")[3].rstrip()
    class_name =  all_files[i].split(" ")[2]
    for j in range(1,len(sirm)):
        f_name = sirm[j][0]
        if file_name == f_name:
            age = sirm[j][4]
            sex = sirm[j][5]
            print(f_name)
            details_sirm.append([all_files[i].split(" ")[1], source, class_name, age,sex])
            break
#%%
import json
path = './pneumonia-challenge-dataset-mappings_2018.json'
f = open(path) 
mapping_json = json.load(f)


mapping = [] 
for idx, sub in enumerate(mapping_json, start = 0): 
    if idx == 0: 
        mapping.append(list(sub.keys())) 
        mapping.append(list(sub.values())) 
    else: 
        mapping.append(list(sub.values())) 


details_rsna = []
for i in range(0,len(all_files)):
    file_name = all_files[i].split(" ")[1]
    source = all_files[i].split(" ")[3].rstrip()
    class_name =  all_files[i].split(" ")[2]
    for j in range(1,len(mapping)):
        f_name = mapping[j][1]+'.png'
        if file_name == f_name:
            map_name = mapping[j][0] 
            print(map_name)
            for k in range(1,len(rsna)):
                fi_name = rsna[k][0]
                if map_name == fi_name:
                    age = rsna[k][4]
                    sex = rsna[k][5]
                    print(file_name)
                    details_rsna.append([file_name, source, class_name, age,sex])
                    break
 #%%
import re
details = details_cohen+details_fig1+details_actmed+details_sirm+details_rsna
age_normal = []
age_pneumonia = []
age_covid = []
sex_normal = []
sex_pneumonia = []
sex_covid = []
for i in range(0,len(details)):
    if details[i][2] == 'normal':
        age_normal.append(int(re.sub("[^0-9]","",details[i][3]) or 0))
        sex_normal.append(details[i][4])
    elif details[i][2] == 'pneumonia':
        age_pneumonia.append(int(re.sub("[^0-9]","",details[i][3]) or 0))
        sex_pneumonia.append(details[i][4])
    elif details[i][2] == 'COVID-19':
        age_covid.append(int(re.sub("[^0-9]","",details[i][3]) or 0))
        sex_covid.append(details[i][4])
age_covid =  list(filter(lambda a: a != 0, age_covid))
age_normal =  list(filter(lambda a: a != 0, age_normal))
age_pneumonia =  list(filter(lambda a: a != 0, age_pneumonia))
sex_covid =  list(filter(lambda a: a != '', sex_covid))
sex_normal =  list(filter(lambda a: a != '', sex_normal))
sex_pneumonia =  list(filter(lambda a: a != '', sex_pneumonia))
#%% age distribution
import matplotlib.pyplot as plt
kwargs = dict(alpha=1, bins=20, density=False, stacked=True,histtype = 'stepfilled')
color1 = [  '#C1CD97','#E18D96','#78A2CC']
color2 = [  '#50874d','#A02C2D','#3065ac']
# Plot
fig = plt.figure()
plt.hist(age_normal, **kwargs, color=color1[0],edgecolor = color2[0], label='Normal')
plt.hist(age_pneumonia, **kwargs, color=color1[1],edgecolor = color2[1], label='Pneumonia')
plt.hist(age_covid, **kwargs, color=color1[2],edgecolor = color2[2], label='COVID-19')
plt.gca().set(title='', ylabel='No of Occurrences',xlabel = 'Age (years)')
#plt.ylim(0,0.1)
plt.legend()


#%% gender distribution 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

color1 = [  '#C1CD97','#E18D96','#78A2CC']
color2 = [  '#50874d','#A02C2D','#3065ac']

labels = ['Normal', 'Pneumonia', 'COVID-19']
men = [sex_normal.count('M'), sex_pneumonia.count('M'), sex_covid.count('M')]
women = [sex_normal.count('F'), sex_pneumonia.count('F'), sex_covid.count('F')]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men, width, label='Male',color=color1[0],edgecolor = color2[0],)
rects2 = ax.bar(x + width/2, women, width, label='Female',color=color1[1],edgecolor = color2[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
#ax.set_title('Gender distribution of three classes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#%% fold wisd class distribution
txt_file_path = './chestxray1/txtfiles/'
            
def rotate(l, n):
    return l[n:] + l[:n]

no_normal = []
no_pneumonia = []
no_covid = []
for k in range(0,5):
    class_category = []
    fold_set = list(range(0,5))
    fold_set = rotate(fold_set, k)
    fold = fold_set[0]
    with open(txt_file_path+'fold_'+str(fold+1)+'.txt', 'r') as fr:
         files = fr.readlines()
    for i in range(0,len(files)):
         class_category.append(files[i].split(" ")[2])
    
    no_normal.append(class_category.count('normal'))
    no_pneumonia.append(class_category.count('pneumonia'))
    no_covid.append(class_category.count('COVID-19'))


labels = ['Fold#1', 'Fold#2', 'Fold#3','Fold#4', 'Fold#5']

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, no_normal, width, label='Normal',color = (0.18, 0.3, 0.3))
rects2 = ax.bar(x, no_pneumonia, width, label='Pneumonia',color = (0, 0.5, .5))
rects3 = ax.bar(x + width, no_covid, width, label='COVID-19',color = (0.4, 0.8, .67))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Class distribution of samples in each fold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), shadow=True, ncol=3)
#%%
import pickle
details_age = details_actmed+details_cohen+details_fig1+details_rsna+details_sirm

with open("./chestxray1/txtfiles/age_gender_details.txt", "wb") as fp:   #Pickling
    pickle.dump(details_age, fp)