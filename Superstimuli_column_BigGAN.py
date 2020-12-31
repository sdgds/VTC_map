# -*- coding: utf-8 -*-
import copy
import cv2
import nibabel as nib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import scipy.stats as stats
from PIL import Image, ImageDraw
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('D:\\TDCNN')
import BrainSOM
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, file_utils)



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
                             std = [0.229, 0.224, 0.225])]),
    'commen': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    }
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet = alexnet.cuda()
alexnet.eval()

Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM/Data.npy')
pca = np.load('D:\\TDCNN\pca_imgnet_50w_mod1.pkl', allow_pickle=True)

som = BrainSOM.VTCSOM(200, 200, 4, sigma=5, learning_rate=1, 
                      neighborhood_function='gaussian', random_seed=None)
som._weights = np.load(r'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_8.0.npy')

biggan = torch.load(r'D:\TDCNN\GAN_models\biggan-deep-128.pkl')
biggan = biggan.cuda()



"""Column segmentation"""
###############################################################################
###############################################################################
def sigmoid(data):
    return 1/(1+np.exp(-5*(data-0.4)))

def column_segmentation(threshold):
    U = som.U_avg_matrix()
    #U = sigmoid(U)
    diff_map = np.zeros((som._x,som._y))
    for i in range(som._x):
        for j in range(som._y):
            dist = U[i,j]
            D = []
            if i-1>=0:
                D.append(U[i-1,j])
            if j-1>=0:
                D.append(U[i,j-1])
            if i+1<=som._x-1:
                D.append(U[i+1,j])
            if j+1<=som._y-1:
                D.append(U[i,j+1])
            D = np.mean(D)
            if (i==0) | (j==0) | (i==som._x-1) | (j==som._y-1):
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
    plt.show()
    return diff_map

def Column_position_set(diff_map):
    columns = {0:[]}
    for i in tqdm(range(1,som._x)):
        for j in range(1,som._y):
            if diff_map[i,j]==0:
                pass
            if diff_map[i,j]==1:
                in_old_column = None
                for col in columns.keys():
                    for position in columns[col]:
                        if position in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                            columns[col].append((i,j))
                            in_old_column = 'yes'
                if in_old_column == None:
                    columns[col+1] = [(i,j)]
                else:
                    pass
    for k in columns.keys():
        columns[k] = list(set(columns[k]))
    columns_set = {0:[]}
    for v in columns.values():
        temp = None
        for col_index in columns_set.keys():
            if set(v) & set(columns_set[col_index]) == set():
                pass
            else:
                temp = 1
                for v_ in v:
                    columns_set[col_index].append(v_)
                columns_set[col_index] = list(set(columns_set[col_index]))
        if temp == None:
            columns_set[col_index+1] = v
    return columns_set
                
def search_column_from_seed(columns_position_set, seed):
    is_seed_in_column = None
    for k in columns_position_set.keys():
        if seed in columns_position_set[k]:
            is_seed_in_column = 1
            temp = np.zeros((som._x,som._y))
            for eliment in columns_position_set[k]:
                temp[eliment] = 1
            plt.figure(figsize=(7,7))
            plt.imshow(temp, cmap='jet')
            return columns_position_set[k], temp
    if is_seed_in_column==None:
        return None, None
        
  
diff_map = column_segmentation(0.01)
columns_position_set = Column_position_set(diff_map)
column_position,pos = search_column_from_seed(columns_position_set, (135,8))





"""Superstimuli"""
###############################################################################
###############################################################################
column_position,pos = search_column_from_seed(columns_position_set, (135,8))
 
W_column = [] 
for x in range(som._x):
    for y in range(som._y):
        if (x,y) in column_position:
            W_column.append(som._weights[x,y,:])
W_column = np.array(W_column).mean(axis=0)
W = np.zeros(1000)
W[:4] = W_column
W_column = pca.inverse_transform(W)

torch.cuda.empty_cache()
batch_size = 10
truncation = 0.3
#threshold = 10
#W = np.where((W_column>=np.percentile(W_column,100-threshold)) | (W_column<=np.percentile(W_column,threshold)), W_column, 0)
W = W_column
weights_vector = torch.Tensor(np.tile(W.reshape(1,1000),(batch_size,1)))
class_vector = weights_vector.cuda()

noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
noise_vector = torch.Tensor(noise_vector).cuda()
noise_vector.requires_grad_(True)
optimizer = torch.optim.Adam([noise_vector], lr=0.1, betas=(0.9,0.999), weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
Correlation = []
for t in tqdm(range(100)):
    output = biggan(noise_vector, class_vector, truncation)
    output = (output-output.min())/(output.max()-output.min())
    inp = torch.nn.functional.interpolate(output, (224,224))
    out = alexnet(inp)
    r_column = torch.cosine_similarity(out,torch.Tensor(W_column).unsqueeze(0).cuda())
    loss = -r_column
    Correlation.append(r_column.max().item())
    optimizer.zero_grad()
    loss.backward(gradient=torch.ones(batch_size).cuda())
    optimizer.step()
    scheduler.step()

output = output[r_column.argmax().item(),:,:,:]
output = (output-output.min())/(output.max()-output.min())
inp = output.cpu()
RF = inp.permute(1,2,0).data.numpy()

plt.figure()
plt.plot(np.array(Correlation))
plt.show()
plt.figure()
plt.imshow(RF, cmap='jet')
plt.show()

act = som.forward_activate(pca.transform(alexnet(inp.unsqueeze(0).cuda()).cpu().data.numpy())[0,[0,1,2,3]])
plt.figure()
plt.imshow(act, cmap='jet')
plt.colorbar()
plt.imshow(pos, alpha=0.5)
plt.show()





"""批量生产j解码Column"""
###############################################################################
###############################################################################
def Superstimuli_on_Column(Column_img, column_position, RF, corelation):
    Superstimuli_img = Image.fromarray(np.uint8(RF*255)).convert('RGB')
    Superstimuli_img = Superstimuli_img.resize((128,128))
    r = []
    c = []
    for i in range(len(column_position)):
        r.append(column_position[i][0])
        c.append(column_position[i][1])
    r_top = min(r)
    c_left = min(c)
    paste_pos = (30*c_left, 30*r_top)
    Column_img.paste(Superstimuli_img, paste_pos)
    draw = ImageDraw.Draw(Column_img)
    draw.text(paste_pos, str(round(corelation,2)), fill="#ff0000")
    return Column_img
   

Columns_position = list(columns_position_set.values())
will_be_remove = []
for column_position in Columns_position:
    if len(column_position)<4:
        will_be_remove.append(column_position)
for remove_col in will_be_remove:
    Columns_position.remove(remove_col)


batch_size = 1
truncation = 0.3
threshold = 10
Column_img = Image.fromarray(np.uint8(diff_map*255)).convert('RGB')
Column_img = Column_img.resize((1920,1920))
Column_cam = copy.deepcopy(Column_img)
for column_position in Columns_position:     
    W_column = [] 
    for (x,y) in column_position:
        W_column.append(som._weights[x,y,:])
    W_column = np.array(W_column).mean(axis=0)
    W = np.zeros(1000)
    W[:4] = W_column
    W_column = pca.inverse_transform(W)
    W_column = np.where(W_column>np.percentile(W_column,99.7), W_column, 0)
    W_column = (W_column-W_column.min())/(W_column.max()-W_column.min())
    
    torch.cuda.empty_cache()
    W = np.where((W_column>=np.percentile(W_column,100-threshold)) | (W_column<=np.percentile(W_column,threshold)), W_column, 0)
    weights_vector = torch.Tensor(np.tile(W.reshape(1,1000),(batch_size,1)))
    class_vector = weights_vector.cuda()
    
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
    noise_vector = torch.Tensor(noise_vector).cuda()
    noise_vector.requires_grad_(True)
    optimizer = torch.optim.SGD([noise_vector], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=250**(-1/3))
    Correlation = []
    for t in tqdm(range(100)):
        output = biggan(noise_vector, class_vector, truncation)
        output = (output-output.min())/(output.max()-output.min())
        inp = torch.nn.functional.interpolate(output, (224,224))
        out = alexnet(inp)
        r_column = torch.cosine_similarity(out,torch.Tensor(W_column).unsqueeze(0).cuda())
        loss = -r_column
        Correlation.append(r_column.max().item())
        optimizer.zero_grad()
        loss.backward(gradient=torch.ones(batch_size).cuda())
        optimizer.step()
        scheduler.step()
    
    # RF
    output = output[r_column.argmax().item(),:,:,:]
    output = (output-output.min())/(output.max()-output.min())
    inp = output.cpu()
    RF = inp.permute(1,2,0).data.numpy()
    Column_img = Superstimuli_on_Column(Column_img, column_position, RF, r_column.max().item())

Column_img.save('/nfs/s2/userhome/zhangyiyuan/Desktop/DCNN_SOM/GAN_images/Column_img.jpg')



