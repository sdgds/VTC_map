#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import nibabel as nib
import numpy as np
import torch
import scipy.stats as stats
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import networkx as nx
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
som = BrainSOM.VTCSOM(64, 64, 1000, sigma=4, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=None)
som._weights = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc8_SOM/pca_init.npy')


# fLoc dataset
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

def Functional_map(class_name):  
    f = os.listdir('/Users/mac/Data/HCP_WM/' + class_name)
    f.remove('.DS_Store')
    Response = []
    for index,pic in enumerate(f):
        img = Image.open('/Users/mac/Data/HCP_WM/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        Response.append(1/som.activate(alexnet(picimg).data.numpy()))   # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
        #Response.append(som.forward_activate(alexnet(picimg).data.numpy().reshape(-1)))
    Response = np.array(Response)
    return Response
    
Response_face = Functional_map('face')
Response_face_avg = np.mean(Response_face, axis=0)

Response_place = Functional_map('place')
Response_place_avg = np.mean(Response_place, axis=0)

Response_body = Functional_map('body')
Response_body_avg = np.mean(Response_body, axis=0)

Response_object = Functional_map('object')
Response_object_avg = np.mean(Response_object, axis=0)

### Area size
# Max activation map
def cohen_d(x1, x2):
    s1 = x1.std()
    s1_ = (x1.shape[0]-1)*(s1**2)
    s2 = x2.std()
    s2_ = (x2.shape[0]-1)*(s2**2)
    s_within = np.sqrt((s1_+s2_)/(x1.shape[0]+x2.shape[0]-2))
    return (x1.mean()-x2.mean())/s_within

color_map = {0:['red','face'],
             1:['green','place'],
             2:['gold','body'],
             3:['midnightblue','object']}
Cohend = []
area_size = {0:0, 1:0, 2:0, 3:0}
Orignal_response = [Response_face,
                    Response_place,
                    Response_body,
                    Response_object]
threshold_cohend = 1.7
plt.figure(figsize=(8,8))
plt.title("The threshold of Cohen's d: %s" %threshold_cohend)
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        max_value = np.max([Response_face_avg[i,j], 
                            Response_place_avg[i,j], 
                            Response_body_avg[i,j],
                            Response_object_avg[i,j]])
        index = np.argmax([Response_face_avg[i,j], 
                           Response_place_avg[i,j], 
                           Response_body_avg[i,j],
                           Response_object_avg[i,j]])
        t, p = stats.ttest_ind(Orignal_response[index][:,i,j], 
                         (np.sum(Orignal_response, axis=0)[:,i,j] - Orignal_response[index][:,i,j])/3)
        cohend = cohen_d(Orignal_response[index][:,i,j], 
                         (np.sum(Orignal_response, axis=0)[:,i,j] - Orignal_response[index][:,i,j])/3)
        Cohend.append(cohend)
        if (p < 0.01/4096) and (cohend>threshold_cohend):
            area_size[index] = area_size[index] + 1
            plt.scatter(i, j, color=color_map[index][0])

plt.figure()
plt.plot(np.sort(Cohend), label="Cohen'd")
plt.plot(threshold_cohend * np.ones(len(Cohend)), label='threshold_cohend', linestyle='--')
plt.legend()

print('face area size:', area_size[0])
print('body area size:', area_size[2])
print('place area size:', area_size[1])
print('object area size:', area_size[3])

            

### Area overlap
def a_b_overlap(a, b):
    a_b = a + b
    print(np.where(a_b==2,1,0).sum())
    
def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1


Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
threshold_cohend = 0.5
    
t_map, p_map = stats.ttest_1samp(Response_face, Contrast_respense[0])
face_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Response_face[:,i,j], Contrast_respense[0][i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            face_mask[i,j] = 1     #也可以用cohend填充
print('Area', face_mask.sum())
plt.plot(np.sort(Cohend), color='red', label='face cohen d')

t_map, p_map = stats.ttest_1samp(Response_place, Contrast_respense[1])
place_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Response_place[:,i,j], Contrast_respense[1][i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            place_mask[i,j] = 1
print('Area', place_mask.sum())
plt.plot(np.sort(Cohend), color='green', label='place cohen d')

t_map, p_map = stats.ttest_1samp(Response_body, Contrast_respense[2])
limb_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Response_body[:,i,j], Contrast_respense[2][i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            limb_mask[i,j] = 1
print('Area', limb_mask.sum())
plt.plot(np.sort(Cohend), color='yellow', label='limb cohen d')

t_map, p_map = stats.ttest_1samp(Response_object, Contrast_respense[3])
object_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Response_object[:,i,j], Contrast_respense[3][i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            object_mask[i,j] = 1
print('Area', object_mask.sum())
plt.plot(np.sort(Cohend), color='blue', label='object cohen d')
plt.plot(threshold_cohend*np.ones(len(Cohend)), label='threshold_cohend', linestyle='--')
plt.legend()

    
a_b_overlap(face_mask, place_mask)

plt.figure(figsize=(6,10))
plt.imshow(face_mask, cmap='Reds', alpha=1, label='face')
plt.imshow(place_mask, cmap='Greens',  alpha=0.4, label='place')
plt.imshow(limb_mask, cmap='Oranges',  alpha=0.3, label='limb')
plt.imshow(object_mask, cmap='Blues',  alpha=0.3, label='object')
plt.axis('off')



### Map correlation
Mask = [face_mask, place_mask, limb_mask, object_mask]
Corr_som = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        Corr_som[i,j] = np.corrcoef(Mask[i].reshape(-1), Mask[j].reshape(-1))[0,1]
plt.figure()
plt.imshow(Corr_som, cmap='jet');plt.colorbar()


HCP_data = nib.load('/Users/mac/Desktop/TDCNN/HCP/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask = nib.load('/Users/mac/Desktop/TDCNN/HCP/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==153)|(mask==160)|(mask==154)|(mask==163)|(mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==153+180)|(mask==160+180)|(mask==154+180)|(mask==163+180))[0]
hcp_vtc = np.zeros(91282)
hcp_vtc[vtc_mask] = 1

hcp_face = HCP_data.dataobj[19,:]
hcp_face = hcp_face * hcp_vtc
hcp_face = np.where(hcp_face>=threshold_cohend, 1, 0)

hcp_place = HCP_data.dataobj[20,:]
hcp_place = hcp_place * hcp_vtc
hcp_place = np.where(hcp_place>=threshold_cohend, 1, 0)

hcp_limb = HCP_data.dataobj[18,:]
hcp_limb = hcp_limb * hcp_vtc
hcp_limb = np.where(hcp_limb>=threshold_cohend, 1, 0)

hcp_object = HCP_data.dataobj[21,:]
hcp_object = hcp_object * hcp_vtc
hcp_object = np.where(hcp_object>=threshold_cohend, 1, 0)

Mask = [hcp_face, hcp_place, hcp_limb, hcp_object]
Corr_hcp = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        Corr_hcp[i,j] = np.corrcoef(Mask[i].reshape(-1), Mask[j].reshape(-1))[0,1]
plt.figure()
plt.imshow(Corr_hcp, cmap='jet');plt.colorbar()

corr = np.corrcoef([Corr_som[0,1], Corr_som[0,2], Corr_som[0,3], Corr_som[1,2], Corr_som[1,3], Corr_som[2,3]], 
                   [Corr_hcp[0,1], Corr_hcp[0,2], Corr_hcp[0,3], Corr_hcp[1,2], Corr_hcp[1,3], Corr_hcp[2,3]])
print(corr)



### Functional connection
function_corr = np.corrcoef(som._weights.reshape(-1,1000))
plt.figure()
plt.imshow(function_corr, cmap='jet')
plt.colorbar()

plt.figure()
n,bins,patches = plt.hist(np.where(function_corr.reshape(-1)>0), bins=100)
plt.figure(figsize=(10,7))
plt.scatter(np.log(np.abs(bins[:-1])), np.log(n))
plt.figure()
n,bins,patches = plt.hist(np.where(function_corr.reshape(-1)<0), bins=100)
plt.figure(figsize=(10,7))
plt.scatter(np.log(np.abs(bins[:-1])), np.log(n))
plt.figure()
n,bins,patches = plt.hist(np.abs(function_corr.reshape(-1)), bins=100)
plt.figure(figsize=(10,7))
plt.scatter(np.log(np.abs(bins[:-1])), np.log(n))


def Graph_position(index):
    coordinates = np.unravel_index(index, (64,64))
    vnode = zip(coordinates[0], coordinates[1])
    npos = dict(zip(index, vnode))
    return npos
    
G = nx.Graph()
M = function_corr
M = np.where((M>np.percentile(M,99)) | (M<np.percentile(M,1)), M, 0)
index = np.random.choice(np.arange(0,4096), 3000)
for i in index:
    for j in index:
        G.add_edge(i,j,weight=M[i,j])

#只画绝对值大的权重
e = [(u,v) for (u,v,d) in G.edges(data=True)]
edgewidth = [np.abs(d['weight'])*5 for (u,v,d) in G.edges(data=True)]
pos = Graph_position(index)
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=10, node_shape='.')
nx.draw_networkx_edges(G, pos, width=edgewidth, edge_color='r')

#权重正负区分开来按照大小画
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<0]
edgewidth_positive = [d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] >0]
edgewidth_negative = [np.abs(d['weight']) for (u,v,d) in G.edges(data=True) if d['weight'] <0]
pos = Graph_position(index)
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=10, node_shape='.')
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=edgewidth_positive, edge_color='r')
nx.draw_networkx_edges(G, pos, edgelist=esmall, alpha=0.6, width=edgewidth_negative,
                       edge_color='b', style='dashed')

#权重正负区分开来画两个图
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<0]
edgewidth_positive = [d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] >0]
edgewidth_negative = [np.abs(d['weight']) for (u,v,d) in G.edges(data=True) if d['weight'] <0]
pos = Graph_position(index)
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=10, node_shape='.')
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=edgewidth_positive, edge_color='r')
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=10, node_shape='.')
nx.draw_networkx_edges(G, pos, edgelist=esmall, alpha=1, width=edgewidth_negative,
                       edge_color='b', style='dashed')




### MFS
# Eccentricity
import h5py

f = h5py.File('/Users/mac/Downloads/retinotopy_stimulus/7T_RETEXP_small.hdf5','r')
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
for t in range(100,110):
    img = Image.fromarray(f['stim'][:][t]).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    plt.figure()
    #plt.imshow(1/som.activate(alexnet(picimg).data.numpy()), 'jet')
    plt.imshow(som.forward_activate(alexnet(picimg).data.numpy().reshape(-1)), 'jet')
    plt.colorbar()
    

# Face vs Place                
plt.figure(figsize=(8,8))              
for i in range(64):
    for j in range(64):
        if face_mask[i,j]==1:
            plt.scatter(i, j, color='red')
        if place_mask[i,j]==1:
            plt.scatter(i, j, color='green')


# Animate vs Inanimate
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

'''Animate: '159.people', '060.duck', '056.dog', '087.goldfish', '204.sunflower-101'
   Inanimate: '091.grand-piano-101', '136.mandolin', '146.mountain-bike', '172.revolver-101', '252.car-side-101' '''

def Response(picdir):
    f = os.listdir(picdir)
    for i in f:
        if i[-3:] != 'jpg':
            f.remove(i)
    response = []
    for pic in f:
        img = Image.open(picdir+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        #response.append(1/som.activate(alexnet(picimg).data.numpy())) 
        response.append(som.forward_activate(alexnet(picimg).data.numpy().reshape(-1)))
    return np.array(response)

Animate_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/253.faces-easy-101/')
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/159.people/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/060.duck/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/056.dog/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/087.goldfish/')))
Animate_response_avg = np.array(Animate_response).mean(axis=0)

Inanimate_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/091.grand-piano-101/')
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/136.mandolin/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/146.mountain-bike/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/172.revolver-101/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/252.car-side-101/')))
Inanimate_response_avg = np.array(Inanimate_response).mean(axis=0)

color_map = {0:['red','animate'], 1:['blue','inanimate']}
threshold_cohend = 0.5
Cohend = []
plt.figure(figsize=(8,8))
plt.title("The threshold of Cohen's d: %s" %threshold_cohend)
for i in range(64):
    for j in range(64):
        max_value = np.max([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        index = np.argmax([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        t, p = stats.ttest_ind(Animate_response[:,i,j], Inanimate_response[:,i,j])
        cohend = cohen_d(Animate_response[:,i,j], Inanimate_response[:,i,j])
        Cohend.append(cohend)
        if (p < 0.01/4096) and (np.abs(cohend)>threshold_cohend):
            plt.scatter(i, j, color=color_map[index][0])

plt.figure()
plt.plot(np.sort(Cohend), label='object cohen d')
plt.plot(threshold_cohend*np.ones(len(Cohend)), label='threshold_cohend', linestyle='--')
plt.legend()

# Big vs Small
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

'''Big: '091.grand-piano-101', '023.bulldozer', '187.skyscraper'
   Small: '027.calculator', '054.diamond-ring', '067.eyeglasses' '''

def Response(picdir):
    f = os.listdir(picdir)
    for i in f:
        if i[-3:] != 'jpg':
            f.remove(i)
    response = []
    for pic in f:
        img = Image.open(picdir+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        response.append(1/som.activate(alexnet(picimg).data.numpy())) 
        #response.append(1/som.activate(alexnet.features(picimg).data.numpy().mean((2,3)).reshape(-1)))
    return np.array(response)

Big_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/091.grand-piano-101/')
Big_response = np.vstack((Big_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/023.bulldozer/')))
Big_response = np.vstack((Big_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/187.skyscraper/')))
Big_response_avg = np.array(Big_response).mean(axis=0)

Small_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/027.calculator/')
Small_response = np.vstack((Small_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/054.diamond-ring/')))
Small_response = np.vstack((Small_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/067.eyeglasses/')))
Small_response_avg = np.array(Small_response).mean(axis=0)

color_map = {0:['blue','big'], 1:['red','small']}
threshold_cohend = 0.5
Cohend = []
plt.figure(figsize=(8,8))
plt.title("The threshold of Cohen's d: %s" %threshold_cohend)
for i in range(64):
    for j in range(64):
        max_value = np.max([Big_response_avg[i,j], Small_response_avg[i,j]])
        index = np.argmax([Big_response_avg[i,j], Small_response_avg[i,j]])
        t, p = stats.ttest_ind(Big_response[:,i,j], Small_response[:,i,j])
        cohend = cohen_d(Big_response[:,i,j], Small_response[:,i,j])
        Cohend.append(cohend)
        if (p < 0.01/4096) and (np.abs(cohend)>threshold_cohend):
            plt.scatter(i, j, color=color_map[index][0])

plt.figure()
plt.plot(np.sort(Cohend), label='object cohen d')
plt.plot(threshold_cohend*np.ones(len(Cohend)), label='threshold_cohend', linestyle='--')
plt.legend()
 
                       
            
### Nested spacial
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

'''Animate: '253.faces-easy-101', '159.people', '060.duck', '056.dog', '087.goldfish'
   Face: fLoc
   Body: fLoc '''

def cohen_d(x1, x2):
    s1 = x1.std()
    s1_ = (x1.shape[0]-1)*(s1**2)
    s2 = x2.std()
    s2_ = (x2.shape[0]-1)*(s2**2)
    s_within = np.sqrt((s1_+s2_)/(x1.shape[0]+x2.shape[0]-2))
    return (x1.mean()-x2.mean())/s_within

def Response(picdir):
    f = os.listdir(picdir)
    for i in f:
        if i[-3:] != 'jpg':
            f.remove(i)
    response = []
    for pic in f:
        img = Image.open(picdir+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        response.append(1/som.activate(alexnet(picimg).data.numpy())) 
    return np.array(response)

Animate_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/253.faces-easy-101/')
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/159.people/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/060.duck/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/056.dog/')))
Animate_response = np.vstack((Animate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/087.goldfish/')))
Animate_response_avg = np.array(Animate_response).mean(axis=0)

Inanimate_response = Response('/Users/mac/Data/caltech256/256_ObjectCategories/091.grand-piano-101/')
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/136.mandolin/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/146.mountain-bike/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/172.revolver-101/')))
Inanimate_response = np.vstack((Inanimate_response, Response('/Users/mac/Data/caltech256/256_ObjectCategories/252.car-side-101/')))
Inanimate_response_avg = np.array(Inanimate_response).mean(axis=0)


plt.figure(figsize=(8,8))
threshold_cohend = 0.5
plt.title("The threshold of Cohen's d: %s" %threshold_cohend)

color_map = {0:['red','animate'], 1:['blue','inanimate']}
for i in range(64):
    for j in range(64):
        max_value = np.max([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        index = np.argmax([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        t, p = stats.ttest_ind(Animate_response[:,i,j], Inanimate_response[:,i,j])
        cohend = cohen_d(Animate_response[:,i,j], Inanimate_response[:,i,j])
        Cohend.append(cohend)
        if (p < 0.01/4096) and (np.abs(cohend)>threshold_cohend) and index==0:
            plt.scatter(i, j, color=color_map[index][0])
            
for i in range(64):
    for j in range(64):
        if face_mask[i,j]==1:
            plt.scatter(i, j, color='black', marker='x', alpha=0.8)
        if limb_mask[i,j]==1:
            plt.scatter(i, j, color='black', marker='.', alpha=0.8)


