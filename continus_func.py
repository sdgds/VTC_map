#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import cv2
import nibabel as nib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import scipy.stats as stats
import PIL.Image as Image
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('/Users/mac/Desktop/TDCNN')
import BrainSOM


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
som = BrainSOM.VTCSOM(64, 64, 1000, sigma=4, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=None)
som._weights = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc8_SOM/pca_init.npy')

Data = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc8_SOM/Data.npy')

Data_fLoc = []
Validation = os.listdir('/Users/mac/Data/fLoc_stim/all_stim')
for Classes in tqdm(Validation):
    pic = '/Users/mac/Data/fLoc_stim/all_stim/'+Classes
    img = Image.open(pic).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    Data_fLoc.append(alexnet(picimg).data.numpy().reshape(-1))
Data_fLoc = np.array(Data_fLoc)
        
Data_WM = []
Validation = os.listdir('/Users/mac/Desktop/TDCNN/HCP/HCP_TFMRI_scripts/WM/WM Stimuli')
Validation.remove('.DS_Store')
for Classes in tqdm(Validation):
    pic = '/Users/mac/Desktop/TDCNN/HCP/HCP_TFMRI_scripts/WM/WM Stimuli/'+Classes
    img = Image.open(pic).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    Data_WM.append(alexnet(picimg).data.numpy().reshape(-1))
Data_WM = np.array(Data_WM)
        


def sigmoid(x):
    return 1/(1+np.exp(-10*(x-0.5)))

pca = PCA()
pca.fit_transform(Data)

theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for x in np.arange(-1,1.01,0.05):
    for y in np.arange(-1,1.01,0.05):
        if y>0 and x>0:
            theta = np.arctan(y/x)
        if y>0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x>0:
            theta = 2*np.pi + np.arctan(y/x)     
        d = np.abs(theta_range-theta)
        r = np.sqrt(x**2+y**2)
        if r < 1:
            plt.scatter(x, y, color=color_map(np.where(d==d.min())[0]))
plt.axis('off')


## linear combination of 5 PC can support som._weights
# how many PC can support som._weights
Correlation = np.zeros((64,64))
for i in range(64):
    for j in range(64):  
        linreg = LinearRegression()
        model = linreg.fit(pca.components_[0:20].T,
                           som._weights[i,j,:].reshape(1000,1))
        Correlation[i,j] = np.corrcoef(np.dot(model.coef_[0],pca.components_[0:20]), som._weights[i,j,:])[0,1]
plt.imshow(Correlation, norm=Normalize(0,1), cmap='jet');plt.colorbar()

# what meaning of PCs?
for pc in range(5):
    Correlation = np.zeros((64,64))
    for i in range(64):
        for j in range(64):  
            linreg = LinearRegression()
            model = linreg.fit(pca.components_[[pc]].T,
                               som._weights[i,j,:].reshape(1000,1))
            Correlation[i,j] = np.corrcoef(np.dot(model.coef_[0],pca.components_[[pc]]), som._weights[i,j,:])[0,1]    
    plt.figure()
    plt.title('Correlation of PC%d' % (pc+1))
    plt.imshow(Correlation, norm=Normalize(0,1), cmap='jet')
    plt.colorbar()

# neuron
R = np.zeros((64,64))
Vector = np.zeros((64,64,5))
Correlation = np.zeros((64,64))
for i in range(64):
    for j in range(64):  
        linreg = LinearRegression()
        model = linreg.fit(np.vstack((pca.components_[0],
                                      pca.components_[1],
                                      pca.components_[2],
                                      pca.components_[3],
                                      pca.components_[4])).T,
                           som._weights[i,j,:].reshape(1000,1))
        x,y,z,p,q = model.coef_[0]
        R[i,j] = np.sqrt(x**2+y**2+z**2+p**2+q**2)
        Vector[i,j,:] = [x,y,z,p,q]
        Correlation[i,j] = np.corrcoef(x*pca.components_[0] + y*pca.components_[1] + z*pca.components_[2] + p*pca.components_[3] + q*pca.components_[4], 
                                       som._weights[i,j,:])[0][1]
plt.imshow(Correlation, norm=Normalize(0,1), cmap='jet');plt.colorbar()

# column
R = np.zeros((64,64))
Vector = np.zeros((64,64,5))
Correlation = np.zeros((64,64))
diff_map = column_segmentation(0.008)
for i in range(64):
    for j in range(64):  
        column_position = column_pos(diff_map, (i,j))
        if column_position!=None:
            W_column = []  
            for x,y in column_position:
                W_column.append(som._weights[x,y,:])
            W_column = np.array(W_column)
            W_column = np.mean(W_column, axis=0)
            linreg = LinearRegression()
            model = linreg.fit(np.vstack((pca.components_[0],
                                          pca.components_[1],
                                          pca.components_[2],
                                          pca.components_[3],
                                          pca.components_[4])).T,
                               som._weights[i,j,:].reshape(1000,1))
            x,y,z,p,q = model.coef_[0]
            R[i,j] = np.sqrt(x**2+y**2+z**2+p**2+q**2)
            Vector[i,j,:] = [x,y,z,p,q]
            Correlation[i,j] = np.corrcoef(x*pca.components_[0] + y*pca.components_[1] + z*pca.components_[2] + p*pca.components_[3] + q*pca.components_[4], 
                                           som._weights[i,j,:])[0][1]
plt.imshow(Correlation, norm=Normalize(0,1), cmap='jet');plt.colorbar()


## neuron level
# two PC
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for i in range(64):
    for j in range(64):  
        linreg = LinearRegression()
        model = linreg.fit(np.vstack((pca.components_[0],
                                      pca.components_[1],
                                      pca.components_[2],
                                      pca.components_[3],
                                      pca.components_[4])).T,
                           som._weights[i,j,:].reshape(1000,1))
        x,y,z,p,q = model.coef_[0]
        a,b = p,q
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        plt.scatter(i, j, marker=',',
                    color=color_map(np.where(d==d.min())[0]), 
                    alpha=1)

# max explanation two PC
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
Corr = np.zeros((64,64))
plt.figure(figsize=(8,8))
for i in range(64):
    for j in range(64):  
        linreg = LinearRegression()
        model = linreg.fit(np.vstack((pca.components_[0],
                                      pca.components_[1],
                                      pca.components_[2],
                                      pca.components_[3],
                                      pca.components_[4])).T,
                           som._weights[i,j,:].reshape(1000,1))
        x,y,z,p,q = model.coef_[0]
        temp = [x,y,z,p,q]
        combine = [(x,y),(x,z),(x,p),(x,q),(y,z),(y,p),(y,q),(z,p),(z,q),(p,q)]
        r = []
        for a,b in combine:
            r.append(np.corrcoef(a*pca.components_[temp.index(a)] + b*pca.components_[temp.index(b)], 
                                som._weights[i,j,:])[0][1])
        Corr[i,j] = np.max(r)
        a,b = combine[np.argmax(r)]
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        plt.scatter(i, j, marker=',',
                    color=color_map(np.where(d==d.min())[0]), 
                    alpha=1)
        
        
## column level
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
diff_map = column_segmentation(0.008)
plt.figure(figsize=(8,8))
for i in range(64):
    for j in range(64):  
        linreg = LinearRegression()
        model = linreg.fit(np.vstack((pca.components_[0],
                                      pca.components_[1],
                                      pca.components_[2],
                                      pca.components_[3],
                                      pca.components_[4])).T,
                           som._weights[i,j,:].reshape(1000,1))
        x,y,z,p,q = model.coef_[0]
        a,b = x,y
        if b>0 and a>0:
            theta = np.arctan(b/a)
        if b>0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a<0:
            theta = np.pi + np.arctan(b/a)
        if b<0 and a>0:
            theta = 2*np.pi + np.arctan(b/a) 
        d = np.abs(theta_range-theta)
        if diff_map[i,j]==1:
            plt.scatter(i, j, marker=',',
                        color=color_map(np.where(d==d.min())[0]), 
                        alpha=1)   
        else:
            plt.scatter(i, j, marker=',',
                        color='white', 
                        alpha=1)  


### HCP_WM
pca = PCA()
pca.fit_transform(Data_WM)

theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.nipy_spectral(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
R = np.zeros((64,64))
Theta = np.zeros((64,64))
Correlation = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        Corr = dict()
        for a in np.arange(-1,1.01,0.05):
            for b in np.arange(-1,1.01,0.05):
                c = a*pca.components_[0] + b*pca.components_[1]
                Corr[np.corrcoef(c, som._weights[i,j,:])[0][1]] = (a,b)  
        Correlation[i,j] = max(Corr.keys())
        x,y = Corr[max(Corr.keys())]
        r = np.sqrt(x**2+y**2)
        R[i,j] = r
        if y>0 and x>0:
            theta = np.arctan(y/x)
        if y>0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x>0:
            theta = 2*np.pi + np.arctan(y/x) 
        Theta[i,j] = theta
        d = np.abs(theta_range-theta)
        plt.scatter(i, j, marker=',',
                    color=color_map(np.where(d==d.min())[0]), 
                    alpha=max(Corr.keys()))

R = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        Corr = dict()
        for a1 in np.arange(-1,1.01,0.1):
            for a2 in np.arange(-1,1.01,0.1):
                for a3 in np.arange(-1,1.01,0.1):
                    c = a1*pca.components_[0] + a2*pca.components_[1] + a3*pca.components_[2]
                    Corr[np.corrcoef(c, som._weights[i,j,:])[0][1]] = (a1,a2,a3)   
        x,y,z = Corr[max(Corr.keys())]
        r = np.sqrt(x**2+y**2+z**2)
        R[i,j] = r


### fLoc
pca = PCA()
pca.fit_transform(Data_fLoc)

theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.rainbow(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
R = np.zeros((64,64))
Theta = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        Corr = dict()
        for a in np.arange(-1,1.01,0.1):
            for b in np.arange(-1,1.01,0.1):
                c = a*pca.components_[0] + b*pca.components_[1]
                Corr[np.corrcoef(c, som._weights[i,j,:])[0][1]] = (a,b)   
        x,y = Corr[max(Corr.keys())]
        r = np.sqrt(x**2+y**2)
        R[i,j] = r
        if y>0 and x>0:
            theta = np.arctan(y/x)
        if y>0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x>0:
            theta = 2*np.pi + np.arctan(y/x) 
        Theta[i,j] = theta
        d = np.abs(theta_range-theta)
        plt.scatter(i, j, marker=',',
                    color=color_map(np.where(d==d.min())[0]), 
                    alpha=r/1.42)
        



"""Gradient"""   
###############################################################################
############################################################################### 
from mpl_toolkits.mplot3d import Axes3D
 
def Laplacian_matrix(som, threshold):
    W = np.corrcoef(som._weights.reshape(-1,1000))
    W = np.where(W>0.9, W, 0)
#    for neuron_i in tqdm(range(4096)):
#        for neuron_j in range(4096):
#            d = np.linalg.norm(W[neuron_i]-W[neuron_j])**2
#            if d<threshold:
#                pass
#            if d>=threshold:
#                W[neuron_i,neuron_j] = 0
    D = np.zeros(W.shape)
    for i in range(4096):
        D[i,i] = W[i].sum()
    L = D - W
    return D,L

D,L = Laplacian_matrix(som, 100)
DL = np.dot(np.linalg.inv(D),L)
eigvalues,eigvectors = np.linalg.eig(DL)
plt.plot(eigvalues)
plt.imshow(eigvectors)

Eigenmap = eigvectors[:,np.argsort(eigvalues)[1]].real.reshape(64,64)

plt.figure()
plt.imshow(Eigenmap, cmap='rainbow')
plt.colorbar()
        

sobelx = cv2.Sobel(Eigenmap, cv2.CV_64F, dx=1, dy=0)
sobely = cv2.Sobel(Eigenmap, cv2.CV_64F, dx=0, dy=1)
result = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
plt.figure()
plt.imshow(result)
plt.colorbar()

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
xx = np.arange(0,64,1)
yy = np.arange(0,64,1)
X, Y = np.meshgrid(xx, yy)
Z = result
surf = ax.plot_surface(X,Y,Z,cmap='jet')
fig.colorbar(surf)


x,y = np.meshgrid(np.linspace(0,64,64),np.linspace(0,64,64))
u = sobelx
v = -sobely
plt.figure(figsize=(7,7))
plt.quiver(x,y,u,v)
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(Eigenmap, cmap='rainbow')
plt.colorbar()
plt.quiver(x,y,u,v)

plt.imshow(cv2.Sobel(sobely, cv2.CV_64F, dx=1, dy=0)-cv2.Sobel(sobelx, cv2.CV_64F, dx=0, dy=1))
plt.colorbar()

U = som.U_min_matrix()
plt.figure(figsize=(10,10))
plt.imshow(U, cmap='bone_r')
plt.colorbar()
plt.quiver(x,y,u,v)




"""Functional column"""   
###############################################################################
###############################################################################   
import copy
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, file_utils)
model = torch.load('/Users/mac/Models/BigGAN_128/biggan-deep-128.pkl')
model = BigGAN.from_pretrained('biggan-deep-256', cache_dir='/Users/mac/Models/BigGAN_256')
model = torch.load('/Users/mac/Models/BigGAN_512/biggan-deep-512.pkl')

## Column segmentation
def column_segmentation(threshold):
    U = som.U_min_matrix()
    diff_map = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            dist = U[i,j]
            D = []
            if i-1>=0:
                D.append(U[i-1,j])
            if j-1>=0:
                D.append(U[i,j-1])
            if i+1<=63:
                D.append(U[i+1,j])
            if j+1<=63:
                D.append(U[i,j+1])
            D = np.mean(D)
            if (i==0) | (j==0) | (i==63) | (j==63):
                diff_map[i,j] = 0
            else:
                diff_map[i,j] = D-dist
    diff_map = np.where(diff_map>threshold,1,0)
    plt.figure(figsize=(7,7))
    plt.imshow(diff_map, cmap='jet')
    plt.axis('off')
    plt.figure(figsize=(7,7))
    plt.imshow(U, cmap=plt.get_cmap('bone_r'))
    plt.imshow(diff_map, cmap='jet', alpha=0.2)
    return diff_map

def column_pos(diff_map, seed):
    x,y = seed
    if diff_map[x,y]==1:
        search = [(x,y)]
        for t in range(100):
            temp = copy.deepcopy(search)
            for point in search:
                if point[0]-1>=0:
                    if diff_map[point[0]-1, point[1]]==1:
                        search.append((point[0]-1, point[1]))
                if point[0]+1>=0:
                    if diff_map[point[0]+1, point[1]]==1:
                        search.append((point[0]+1, point[1]))
                if point[1]-1>=0:
                    if diff_map[point[0], point[1]-1]==1:
                        search.append((point[0], point[1]-1))
                if point[1]+1>=0:
                    if diff_map[point[0], point[1]+1]==1:
                        search.append((point[0], point[1]+1))
                search = list(set(search))
            if len(search)==len(temp):
                break
        pos = np.zeros((64,64))
        for index in search:
            pos[index] = 1
        plt.figure(figsize=(7,7))
        plt.imshow(pos, cmap='jet')
        #plt.axis('off') 
        return search, pos
    else:
        return None
    
diff_map = column_segmentation(0.0035)
column_position,pos = column_pos(diff_map, (30,32))
 
W_column = [] 
W_not_column = []
for x in range(64):
    for y in range(64):
        if (x,y) in column_position:
            W_column.append(som._weights[x,y,:])
        else:
            W_not_column.append(som._weights[x,y,:])
W_column = np.array(W_column).mean(axis=0)
W_not_column = np.array(W_not_column).mean(axis=0)

R = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        R[i,j] = np.corrcoef(som._weights[i,j,:],W_column)[0,1]
plt.figure(figsize=(7,7))
plt.imshow(R, cmap='jet', norm=Normalize(0,1))
plt.colorbar()
plt.imshow(pos, alpha=0.5)


## GAN decoding
truncation = 0.3
W = np.where((W_column>=np.percentile(W_column,90)) | (W_column<=np.percentile(W_column,10)), W_column, 0)
weights_vector = torch.Tensor(W.reshape(1,1000))
class_vector = weights_vector
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
noise_vector = torch.from_numpy(noise_vector)

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)
    output = (output-output.min())/(output.max()-output.min())

plt.figure()
plt.imshow(output[0].permute(1,2,0).data.numpy())
plt.axis('off')



transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
img = transforms.ToPILImage()(output[0])
inp = transform(img).unsqueeze(0)*255
optimizer = torch.optim.Adam([inp], lr=0.2)
Correlation = []
for t in range(100):
    inp.requires_grad_(True)
    out = alexnet(inp)  
    r_column = torch.cosine_similarity(out[0].unsqueeze(0), 
                                       torch.Tensor(W_column).unsqueeze(0))
    r_not_column = torch.cosine_similarity(out[0].unsqueeze(0), 
                                           torch.Tensor(W_not_column).unsqueeze(0))
    loss = -(r_column-0.5*r_not_column)# + 0.1*torch.norm(inp)
#    Exp_R = np.zeros((64,64))
#    for x in range(64):
#        for y in range(64):
#            Exp_R[x,y] = torch.exp(torch.cosine_similarity(out[0].unsqueeze(0), 
#                         torch.Tensor(som._weights[x,y,:]).unsqueeze(0)))
#    loss = -(torch.exp(r_column)/Exp_R.sum())
    Correlation.append(r_column.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

RF = inp[0].permute(1,2,0).data.numpy()

plt.figure(figsize=(11,5))
plt.plot(W_column/W_column.max())
plt.plot(out[0].data.numpy()/out.max().item(), alpha=0.7)

plt.figure()
plt.plot(np.array(Correlation))
plt.figure()
plt.imshow(RF/255, cmap='jet')


act = som.activate(alexnet(inp).data.numpy())
plt.figure()
plt.imshow(1/act, cmap='jet')
plt.colorbar()
plt.imshow(pos, alpha=0.5)


