# -*- coding: utf-8 -*-
import copy
import cv2
import itertools
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
sys.path.append('/home/dell/TDCNN/')
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

Data = np.load('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/Data.npy')
pca = PCA()
pca.fit(Data)

som = BrainSOM.VTCSOM(200, 200, 4, sigma=4, learning_rate=1, neighborhood_function='gaussian', random_seed=0)
som._weights = np.load('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/som_sigma_7.7.npy')

biggan = torch.load('/home/dell/TDCNN/GAN_models/biggan-deep-128.pkl')
biggan = biggan.cuda()



"""Column segmentation"""
###############################################################################
###############################################################################
def sigmoid(data):
    return 1/(1+np.exp(-10*(data-0.4)))

def column_segmentation(threshold):
    U = som.U_avg_matrix()
    U_copy = copy.deepcopy(U)
    U = sigmoid(U)
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
    plt.imshow(U_copy, cmap=plt.get_cmap('bone_r'))
    plt.imshow(diff_map, cmap='jet', alpha=0.2)
    plt.show()
    return diff_map

def Column_position_set(diff_map):
    def generative_column(seed_position):
        one_column_pos = set()
        one_column_pos.add(seed_position)
        on_off = 1
        while on_off==1:
            one_column_pos_copy = copy.deepcopy(one_column_pos)
            for seed_pos in one_column_pos:
                for ii in range(seed_pos[0]-1, seed_pos[0]+2):
                    for jj in range(seed_pos[1]-1, seed_pos[1]+2):
                        if (ii >= 0 and ii < som._weights.shape[0] and
                            jj >= 0 and jj < som._weights.shape[1] and diff_map[ii,jj]==1):   
                            one_column_pos_copy.add((ii,jj))
            if one_column_pos_copy==one_column_pos:
                on_off = 0
            else:
                one_column_pos = one_column_pos_copy
        return one_column_pos
    columns_position = zip(np.where(diff_map>0)[0], np.where(diff_map>0)[1])
    columns_pos_list = []
    for seed_position in columns_position:
        if seed_position not in list(itertools.chain.from_iterable(columns_pos_list)):
            columns_pos_list.append(generative_column(seed_position))
    columns_pos_dict = dict()
    for k,v in enumerate(columns_pos_list):
        columns_pos_dict[k] = list(v)
    return columns_pos_dict
                
def search_column_from_seed(columns_pos_dict, seed):
    is_seed_in_column = None
    for k in columns_pos_dict.keys():
        if seed in columns_pos_dict[k]:
            is_seed_in_column = 1
            temp = np.zeros((som._x,som._y))
            for eliment in columns_pos_dict[k]:
                temp[eliment] = 1
            plt.figure(figsize=(7,7))
            plt.imshow(temp, cmap='jet')
            return columns_pos_dict[k], temp
    if is_seed_in_column==None:
        return None, None
        
  
diff_map = column_segmentation(0.001)
columns_pos_dict = Column_position_set(diff_map)
column_position,pos = search_column_from_seed(columns_pos_dict, (17,7))





"""Superstimuli"""
###############################################################################
###############################################################################
column_position,pos = search_column_from_seed(columns_pos_dict, (100,150))

W_column = [] 
W_not_column = []
for x in range(som._x):
    for y in range(som._y):
        if (x,y) in column_position:
            W_column.append(som._weights[x,y,:])
        else:
            W_not_column.append(som._weights[x,y,:])
W_column = np.array(W_column).mean(axis=0)
W_not_column = np.array(W_not_column).mean(axis=0)
W = np.zeros(1000)
W[:4] = W_column
W = pca.inverse_transform(W)

# initial noise
torch.cuda.empty_cache()
batch_size = 10
truncation = 0.3
weights_vector = torch.Tensor(np.tile(W.reshape(1,1000),(batch_size,1)))
class_vector = weights_vector.cuda()
r_column_mean = 0
while r_column_mean<=0.5:
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
    noise_vector = torch.Tensor(noise_vector).cuda()
    output = biggan(noise_vector, class_vector, truncation)
    output = (output-output.min())/(output.max()-output.min())
    inp = torch.nn.functional.interpolate(output, (224,224))
    out = alexnet(inp)
    r_column = torch.cosine_similarity(torch.Tensor(pca.transform(out.cpu().data)[:,:4]).cuda(),
                                       torch.Tensor(W_column).unsqueeze(0).cuda())
    r_column_mean = r_column.mean()

# train
batch_size = 10
truncation = 0.3
weights_vector = torch.Tensor(np.tile(W.reshape(1,1000),(batch_size,1)))
class_vector = weights_vector.cuda()
noise_vector.requires_grad_(True)
optimizer = torch.optim.Adam([noise_vector], lr=0.1, betas=(0.9,0.999), weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
Correlation = []
for t in tqdm(range(100)):
    output = biggan(noise_vector, class_vector, truncation)
    output = (output-output.min())/(output.max()-output.min())
    inp = torch.nn.functional.interpolate(output, (224,224))
    out = alexnet(inp)
    r_column = torch.cosine_similarity(torch.Tensor(pca.transform(out.cpu().data)[:,:4]).cuda(),
                                       torch.Tensor(W_column).unsqueeze(0).cuda())
    loss = -r_column
    loss.requires_grad=True
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

act = 1/som.activate(pca.transform(alexnet(inp.unsqueeze(0).cuda()).cpu().data.numpy())[0,[0,1,2,3]])
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
    paste_pos = (10*c_left, 10*r_top)
    Column_img.paste(Superstimuli_img, paste_pos)
    return Column_img

def Superstimuli_on_Column_with_text(Column_img_text, column_position, RF, corelation):
    Superstimuli_img = Image.fromarray(np.uint8(RF*255)).convert('RGB')
    Superstimuli_img = Superstimuli_img.resize((128,128))
    r = []
    c = []
    for i in range(len(column_position)):
        r.append(column_position[i][0])
        c.append(column_position[i][1])
    r_top = min(r)
    c_left = min(c)
    paste_pos = (10*c_left, 10*r_top)
    Column_img_text.paste(Superstimuli_img, paste_pos)
    draw = ImageDraw.Draw(Column_img_text)
    draw.text(paste_pos, str(round(corelation,2)), fill="#ff0000")
    return Column_img_text
   

Columns_position = list(columns_pos_dict.values())
will_be_remove = []
for column_position in Columns_position:
    if len(column_position)<4:
        will_be_remove.append(column_position)
for remove_col in will_be_remove:
    Columns_position.remove(remove_col)


batch_size = 10
truncation = 0.3
Column_img = Image.fromarray(np.uint8(diff_map*255)).convert('RGB').resize((2000,2000))
Column_img_text = Image.fromarray(np.uint8(diff_map*255)).convert('RGB').resize((2000,2000))
for column_position in tqdm(Columns_position):     
    W_column = [] 
    W_not_column = []
    for x in range(som._x):
        for y in range(som._y):
            if (x,y) in column_position:
                W_column.append(som._weights[x,y,:])
            else:
                W_not_column.append(som._weights[x,y,:])
    W_column = np.array(W_column).mean(axis=0)
    W_not_column = np.array(W_not_column).mean(axis=0)
    W = np.zeros(1000)
    W[:4] = W_column
    W = pca.inverse_transform(W)
    
    torch.cuda.empty_cache()
    weights_vector = torch.Tensor(np.tile(W.reshape(1,1000),(batch_size,1)))
    class_vector = weights_vector.cuda()
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
    noise_vector = torch.Tensor(noise_vector).cuda()
    noise_vector.requires_grad_(True)
    optimizer = torch.optim.Adam([noise_vector], lr=0.1, betas=(0.9,0.999), weight_decay=0.005)
    #optimizer = torch.optim.SGD([noise_vector], lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=250**(-1/3))
    Correlation = []
    for t in range(100):
        output = biggan(noise_vector, class_vector, truncation)
        output = (output-output.min())/(output.max()-output.min())
        inp = torch.nn.functional.interpolate(output, (224,224))
        out = alexnet(inp)
        r_column = torch.cosine_similarity(torch.Tensor(pca.transform(out.cpu().data)[:,:4]).cuda(),
                                           torch.Tensor(W_column).unsqueeze(0).cuda())
        loss = -r_column
        loss.requires_grad = True
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
    Column_img_text = Superstimuli_on_Column_with_text(Column_img_text, column_position, RF, r_column.max().item())

Column_img.save('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/Column_img.jpg')
Column_img_text.save('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/Column_img_text.jpg')


