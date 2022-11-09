import pickle
import os
import numpy as np
import csv
from sklearn import preprocessing

rootdir = '../../yongwei_Data/'
print(rootdir)
fea_name = 'gs_feature'
fea_file = rootdir + fea_name
label_name = 'labels'
label_file = rootdir + label_name

def Get_fea(dir):
    traindata = []
    for sess in os.listdir(dir):
        data_dir = dir + '/' + sess
        data_1 = []
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        num = 0
        for row in file_content:
            data = {}
            x = []
            for i in range(1,len(row)):
                row[i] = float(row[i])
                b = np.isinf(row[i])
                # print(b)
                if b:
                    print(row[i])
                x.append(row[i])
            row_1 = np.array(x)
            data['id'] = sess[:-3] + str(num)
            data['time'] = float(row[0])
            data['fea_data'] = row_1
            num = num + 1
            data_1.append(data)
        traindata.append(data_1)
    print(len(traindata))
    return traindata

def Get_label(root_dir,name_1,name_2):
    dir = root_dir + '/' + name_1
    traindata_1 = []
    for sess in os.listdir(dir):
        print(sess)
        num = 0
        data_dir = dir + '/' + sess
        data_1 = []
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        for row in file_content:
            data = {}
            data['id'] = sess[:-3] + str(num)
            data['label_V'] = row[-1]
            num = num + 1
            data_1.append(data)
        traindata_1.append(data_1)
    print(len(traindata_1))

    dir = root_dir + '/' + name_2
    traindata_2 = []
    for sess in os.listdir(dir):
        print(sess)
        num = 0
        data_dir = dir + '/' + sess
        data_1 = []
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        for row in file_content:
            data = {}
            data['id'] = sess[:-3] + str(num)
            data['label_A'] = row[-1]
            num = num + 1
            data_1.append(data)
        traindata_2.append(data_1)
    print(len(traindata_2))

    for i in range(len(traindata_1)):
        for j in range(len(traindata_1[i])):
            traindata_1[i][j]['label_A'] = traindata_2[i][j]['label_A']
    return traindata_1

def Class_data(fea_data):
    Separate_label = [0,]
    time = 0
    while time <= 300:
        time = time + 0.04
        Separate_label.append(time)
    print(Separate_label)
    for i in range(len(fea_data)):
        for j in range(len(fea_data[i])):
            x = 0
            while (x <= len(Separate_label)-1):
                if(Separate_label[x]<fea_data[i][j]['time']<=Separate_label[x+1]):
                    fea_data[i][j]['time_class'] = x+1
                    #print(x)
                    x = 7501
                x = x+1
            print(j)
    print(fea_data[0][:10])

    return fea_data
all_data = Get_fea(fea_file)
All_label = Get_label(label_file,'valence','aroual')
All_data_class = Class_data(all_data)

for i in range(len(All_label)):
    for j in range(len(All_label[i])):
        fea = []
        for x in range(len(All_data_class[i])):
            if(str(j+1) == str(All_data_class[i][x]['time_class'])):
                fea.append(All_data_class[i][x]['fea_data'])
        All_label[i][j]['ALL_fea_data'] = fea
        print(j)

a = [0.0 for i in range(5)]
a = np.array(a)

lens = []
for i in range(len(All_label)):
    for j in range(len(All_label[i])):
        ha = []
        if(len(All_label[i][j]['ALL_fea_data']) < 5):
            for z in range(len(All_label[i][j]['ALL_fea_data'])):
                ha.append(np.array(All_label[i][j]['ALL_fea_data'][z]))
            len_zero = 5 - len(All_label[i][j]['ALL_fea_data'])
            for x in range(len_zero):
                ha.append(a)
        else:
            for z in range(len(All_label[i][j]['ALL_fea_data'])):
                if(z < 5):
                    ha.append(np.array(All_label[i][j]['ALL_fea_data'][z]))
        All_label[i][j]['ALL_fea_data'] = ha


train_data = []
test_data = []
for i in range(len(All_label)):
    if(All_label[i][0]['id'][0] == 't'):
        train_data.append(All_label[i])
    if (All_label[i][0]['id'][0] == 'd'):
        test_data.append(All_label[i])

print(len(train_data))
print(len(test_data))

print(train_data[0][:5])
print(test_data[0][:5])
file = open('train_data.pickle', 'wb')
pickle.dump(train_data,file)
file.close()
file = open('test_data.pickle', 'wb')
pickle.dump(test_data,file)
file.close()
