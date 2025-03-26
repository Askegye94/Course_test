# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:13:29 2024

@author: sdd380
"""

import os
os.chdir('C:/Users/sdd380/surfdrive - David, S. (Sina)@surfdrive.surf.nl/Projects/SOM_Workshop/ISBS2024_ML/Unsupervised Learning/')
# os.chdir('C:/Users/sdd380/surfdrive - David, S. (Sina)@surfdrive.surf.nl/Projects/SOM_Stroke/')
from som_data_struct import som_data_struct
from som_normalize import som_normalize
from som_make import som_make
from som_bmus import som_bmus
from som_ind2sub import som_ind2sub
from som_denormalize import som_denormalize
from reader import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
seed = 42

## define dataset
filename = r'C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Sanne_Heliomare\DatabaseRYSEN_after.csv'
filename_test = r'C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Sanne_Heliomare\DatabaseRYSEN_before.csv'


# Read data
headers = read_headers(filename, delimiter=',' )
data = read_data(filename, delimiter=',')

### Check for missing values

missing_rows = np.isnan(data)
num_missing_rows = missing_rows.sum()
print(f"Number of rows with missing values: {num_missing_rows}")


imputer = IterativeImputer(max_iter=10, random_state=seed)
data_imputed = imputer.fit_transform(data)


missing_rows = np.isnan(data_imputed)
num_missing_rows = missing_rows.sum()
print(f"New number of rows with missing values: {num_missing_rows}")

 

 
#####
label_header = headers[14]
headers = np.delete(headers,[1,5,7,9,10,11,12,13,14])

label_data = data_imputed[:,14]
data=np.delete(data_imputed,[1,5,7,9,10,11,12,13,14],axis=1)


# test=np.transpose(data.copy())

# create train and test structures
sData = som_data_struct(data.copy())
sData_copy = copy.deepcopy(sData)

# test_data = read_data(filename_test,delimiter=',')
# sTest = som_data_struct(test_data.copy())
# sTest['comp_names'] = headers
# sTest_copy = copy.deepcopy(sTest)

## Normalize the Data
plotdata = sData['data'].copy()
sData_norm = som_normalize(sData_copy, 'var')

# plotdata_test = sTest['data'].copy()
# sTest_norm = som_normalize(sTest_copy, 'var')
# sTest_norm_copy = copy.deepcopy(sTest_norm)


## Train the SOM
sMap = som_make(sData_norm, *['lattice', 'shape', 'training', 'initi'],**{'lattice':'hexa', 'shape':'sheet', 'training': 'default', 'init': 'lininit'})

sMap['comp_names'] = headers
sMap['labels']=label_data


## Find best-matching units
Traj_train, Qerrs_train = som_bmus(sMap, sData_norm, 'all')
Traj_train_coord = som_ind2sub(sMap, Traj_train[:,0])
Traj_train_coord = np.concatenate((Traj_train_coord, Qerrs_train[:, [0]]), axis=1)
line1 = np.concatenate((sMap['topol']['msize'], [0]))

## Denormalize the weight vectors
M = som_denormalize(sMap['codebook'].copy(), *[sMap])

Traj_train, Qerrs_train = som_bmus(M, plotdata.copy(), 'all')
Traj_train_coord = som_ind2sub(sMap, Traj_train[:,0])
Traj_train_coord = np.concatenate((Traj_train_coord, Qerrs_train[:, [0]]), axis=1)

## index all input vectors assigned to one neuron
# find the lines that hit each neuron
index = [[None for _ in range(line1[1])] for _ in range(line1[0])]

# Iterate over t and q using nested loops
for t in range(0, line1[1]):
    for q in range(0, line1[0]):
        index[q][t] = np.where((Traj_train_coord[:, 0] == q) & (Traj_train_coord[:, 1] == t))[0]

# Compute average Frame number per neuron
# Flatten index using list comprehension
index_reQE = [item for sublist in index for item in sublist]



Labels_SOM1 = np.zeros((len(M), 1))
for r in range(len(M)):
    Labels_SOM1[r, 0] = np.sum(1/Traj_train_coord[index_reQE[r],2]*label_data[index_reQE[r]])/np.sum(1/Traj_train_coord[index_reQE[r],2])

Labels_re = Labels_SOM1.reshape(line1[0], line1[1])

extent = [0, Labels_re.shape[0], 0, Labels_re.shape[1]]

plt.figure(figsize=(19, 9))
plt.imshow(Labels_re, aspect='auto', cmap='viridis', origin='lower', extent=extent)
plt.colorbar(label='Rysen yes no (%)')
plt.xlabel('X Coordinate SOM')
plt.ylabel('Y Coordinate SOM')
plt.title('2D Heatmap of Labels together with the Trajectories')


Ntest=np.sum(label_data, dtype=np.int32)
Ntrain=len(Traj_train_coord)-Ntest

### CLUSTER the data based on the bmus
data1 = np.reshape(Traj_train_coord[label_data==0,0:3],(Ntrain, 1,3))
data2 = np.reshape(Traj_train_coord[label_data==1,0:3],(Ntest, 1,3))

# Combine data along the first axis (vertical concatenation)
combined_data = np.concatenate((data1, data2), axis=0)

# Reshape data for clustering
flattened_data = combined_data.reshape(combined_data.shape[0], -1)

# Initialize K-means model
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit K-means model to flattened data
kmeans.fit(flattened_data)

# Get cluster labels
cluster_labels = kmeans.labels_

# Separate indices of data1 and data2
num_data1 = data1.shape[0]
num_data2 = data2.shape[0]

# Create arrays to store indices of data1 and data2 in each cluster
data1_clusters = np.zeros(num_data1, dtype=int)  # Array to store cluster labels for data1
data2_clusters = np.zeros(num_data2, dtype=int)  # Array to store cluster labels for data2

# Assign cluster labels to data1
data1_clusters[cluster_labels[:num_data1] == 0] = 1  # Assign cluster 1 to trajectories in data1
data1_clusters[cluster_labels[:num_data1] == 1] = 2  # Assign cluster 2 to trajectories in data1

# Assign cluster labels to data2
data2_clusters[cluster_labels[num_data1:] == 0] = 1  # Assign cluster 1 to trajectories in data2
data2_clusters[cluster_labels[num_data1:] == 1] = 2  # Assign cluster 2 to trajectories in data2

# Combine the cluster labels into a single array
cluster_assignment = np.concatenate((data1_clusters, data2_clusters))


# Print the cluster assignments (just for demonstration)
print("Cluster assignments:")
for i in range(len(cluster_assignment)):
    if i < num_data1:
        print(f"Data1 slice {i} is in Cluster {cluster_assignment[i]}")
    else:
        print(f"Data2 slice {i - num_data1} is in Cluster {cluster_assignment[i]}")

correlation_matrix = np.corrcoef(cluster_assignment, label_data)
        