#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from tqdm import tqdm
import nibabel as nib
import numpy as np
import torch
import scipy.stats as stats
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import numpy as np
#from dipy.data import get_fnames
#from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
#from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
#import dipy.align.imwarp as imwarp
#from dipy.viz import regtools
from bayes_opt import BayesianOptimization
import sys
sys.path.append('/home/dell/TDCNN/')
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
def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))

def none_decay(scalar, t, max_iter):
    return scalar

som = BrainSOM.VTCSOM(64, 64, 1000, sigma=4, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=None)
som._weights = np.load('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/pca_init.npy')

som = BrainSOM.VTCSOM(64, 64, 4096, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
som._weights = np.load('/home/dell/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_longinh_shortexc.npy')
som.excitory_value = np.load('/home/dell/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_excitory_value.npy')
som.inhibition_value = np.load('/home/dell/TDCNN/Results/Alexnet_fc6_SOM/Alexnet_fc6_pca_init_inhibition_value.npy')


# HCP data representation in Neural network
class NET_fc(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET_fc, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.conv_output = layer_out
        self.model.classifier[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)

alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
model = alexnet
model_truncate = NET_fc(model, selected_layer=2)

def Representation_class(class_name):  
    f = os.listdir('/home/dell/TDCNN/HCP/HCP_WM/' + class_name)
    f.remove('.DS_Store')
    Response = []
    for index,pic in tqdm(enumerate(f)):
        img = Image.open('/home/dell/TDCNN/HCP/HCP_WM/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        model_truncate.layeract(picimg)
        Response.append(model_truncate.conv_output.data.numpy()[0])   
    Response = np.array(Response)
    return Response
    
Representation_face = Representation_class('face')
Representation_place = Representation_class('place')
Representation_body = Representation_class('body')
Representation_object = Representation_class('object')

plt.imshow(np.corrcoef(np.vstack((Representation_face,
                                  Representation_place,
                                  Representation_body,
                                  Representation_object))))
plt.colorbar()


# HCP data test fc6 4096
class NET_fc(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET_fc, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.conv_output = layer_out
        self.model.classifier[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)

alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
model = alexnet
model_truncate = NET_fc(model, selected_layer=2)

def Functional_map(class_name, som):  
    f = os.listdir('/home/dell/TDCNN/HCP/HCP_WM/' + class_name)
    f.remove('.DS_Store')
    Response = []
    for index,pic in tqdm(enumerate(f)):
        img = Image.open('/home/dell/TDCNN/HCP/HCP_WM/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        model_truncate.layeract(picimg)
        Response.append(1/som.activate(model_truncate.conv_output.data.numpy()))   # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
    Response = np.array(Response)
    return Response
    
Response_face = Functional_map('face', som)
Response_face_avg = np.mean(Response_face, axis=0)

Response_place = Functional_map('place', som)
Response_place_avg = np.mean(Response_place, axis=0)

Response_body = Functional_map('body', som)
Response_body_avg = np.mean(Response_body, axis=0)

Response_object = Functional_map('object', som)
Response_object_avg = np.mean(Response_object, axis=0)


# HCP data test last layer 1000
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

def Functional_map(class_name, som):  
    f = os.listdir('/home/dell/TDCNN/HCP/HCP_WM/' + class_name)
    f.remove('.DS_Store')
    Response = []
    for index,pic in tqdm(enumerate(f)):
        img = Image.open('/home/dell/TDCNN/HCP/HCP_WM/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        Response.append(1/som.activate(alexnet(picimg).data.numpy()))   # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
    Response = np.array(Response)
    return Response
    
Response_face = Functional_map('face', som)
Response_face_avg = np.mean(Response_face, axis=0)

Response_place = Functional_map('place', som)
Response_place_avg = np.mean(Response_place, axis=0)

Response_body = Functional_map('body', som)
Response_body_avg = np.mean(Response_body, axis=0)

Response_object = Functional_map('object', som)
Response_object_avg = np.mean(Response_object, axis=0)

            

### Area overlap
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

    
plt.figure(figsize=(6,10))
plt.imshow(face_mask, cmap='Reds', alpha=1, label='face')
plt.imshow(place_mask, cmap='Greens',  alpha=0.4, label='place')
plt.imshow(limb_mask, cmap='Oranges',  alpha=0.3, label='limb')
plt.imshow(object_mask, cmap='Blues',  alpha=0.3, label='object')
plt.axis('off')

plt.figure(figsize=(6,10))
plt.imshow(face_mask, cmap='Reds', alpha=1, label='face');plt.show()
plt.figure(figsize=(6,10))
plt.imshow(place_mask, cmap='Greens',  alpha=0.4, label='place');plt.show()
plt.figure(figsize=(6,10))
plt.imshow(limb_mask, cmap='Oranges',  alpha=0.3, label='limb');plt.show()
plt.figure(figsize=(6,10))
plt.imshow(object_mask, cmap='Blues',  alpha=0.3, label='object');plt.show()





"""Quantitative Similarity"""
###############################################################################
###############################################################################
# Utility
HCP_data = nib.load('/home/dell/TDCNN/HCP/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask = nib.load('/home/dell/TDCNN/HCP/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==153)|(mask==160)|(mask==154)|(mask==163)|(mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==153+180)|(mask==160+180)|(mask==154+180)|(mask==163+180))[0]
hcp_vtc = np.zeros(91282)
hcp_vtc[vtc_mask] = 1
R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==153)|(mask==160)|(mask==154)|(mask==163))[0]
R_hcp_vtc = np.zeros(91282)
R_hcp_vtc[R_vtc_mask] = 1
L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==153+180)|(mask==160+180)|(mask==154+180)|(mask==163+180))[0]
L_hcp_vtc = np.zeros(91282)
L_hcp_vtc[L_vtc_mask] = 1

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


### Purity of area
def Purity_som(area_name, prune=True):
    Response_all = []
    Response_all.append(Response_face.mean(axis=0))
    Response_all.append(Response_place.mean(axis=0))
    Response_all.append(Response_body.mean(axis=0))
    Response_all.append(Response_object.mean(axis=0))
    Response_all = np.array(Response_all)
    if area_name=='face':
        index = 0
    if area_name=='place':
        index = 1
    if area_name=='body':
        index = 2
    if area_name=='object':
        index = 3
    purity = Response_all[index,:,:]-(Response_all.sum(axis=0)-Response_all[index,:,:])/3
    center = np.unravel_index(purity.argmax(), (64,64)) 
    R = []
    for i in range(64):
        for j in range(64):
            r = np.sqrt((i-center[0])**2+(j-center[1])**2)
            R.append(r)
    R = list(set(R))      # All posible radius length
    purity_dict = dict()
    for r in R:
        purity_dict[r] = []     # key is the element of R
    for i in range(64):
        for j in range(64):
            r = np.sqrt((i-center[0])**2+(j-center[1])**2)
            purity_dict[r].append(purity[i,j])      # value is the element of purity
    for k,v in purity_dict.items():
        purity_dict[k] = np.mean(v)        # avrage of values in a key
    purity_dict_prune = dict()
    for k in range(int(max(list(purity_dict.keys())))+1):
        V_ = []
        for k_,v_ in purity_dict.items():
            if 0 <= k_-k <= 1:
                V_.append(v_)
        purity_dict_prune[k] = np.mean(V_)
    if prune==False:
        plt.figure()
        plt.scatter(list(purity_dict.keys()), list(purity_dict.values()), s=3) 
        plt.title('SOM:'+area_name)
    if prune==True:
        plt.figure()
        plt.scatter(list(purity_dict_prune.keys()), list(purity_dict_prune.values()), s=3)
        plt.title('SOM:'+area_name)

Purity_som('face', prune=True)
Purity_som('place', prune=True)
Purity_som('body', prune=True)
Purity_som('object', prune=True)


def get_hcp_space_map(hcp_index, plot=False):
    # geometry information
    geometry = nib.load('/nfs/a2/HCP_S1200/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * hcp_vtc
    # make dict
    list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
    CORTEX_Left = list_of_block[0]
    Dict_hcp_to_32kL = dict()
    for vertex in range(CORTEX_Left.index_count):
        Dict_hcp_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(hcp_vtc[:CORTEX_Left.index_count]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
        hcp_32K.append(hcp_data[i])
    if plot==False:
        pass
    if plot==True:
        # plot
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(geometry[vtc_32K][:,0], geometry[vtc_32K][:,1], marker='.',
                    c=hcp_32K, cmap=plt.cm.jet)
    position = geometry[vtc_32K][:,:2]
    value = hcp_32K
    return position, value

def Purity_hcp(area_name, prune=True):
    position, value_face = get_hcp_space_map(19, plot=False)
    position, value_place = get_hcp_space_map(20, plot=False)
    position, value_body = get_hcp_space_map(18, plot=False)
    position, value_object = get_hcp_space_map(21, plot=False)    # position is the vtc position
    Response_all = []
    Response_all.append(value_face)
    Response_all.append(value_place)
    Response_all.append(value_body)
    Response_all.append(value_object)
    Response_all = np.array(Response_all)
    if area_name=='face':
        index = 0
    if area_name=='place':
        index = 1
    if area_name=='body':
        index = 2
    if area_name=='object':
        index = 3
    purity = Response_all[index,:]-(Response_all.sum(axis=0)-Response_all[index,:])/3
    center = position[purity.argmax()].tolist()
    purity_dict = dict()
    for pos_index,(i,j) in enumerate(position):
        r = np.sqrt((i-center[0])**2+(j-center[1])**2)
        purity_dict[r] = purity[pos_index]
    purity_dict_prune = dict()
    for k in range(int(max(list(purity_dict.keys())))+1):
        V_ = []
        for k_,v_ in purity_dict.items():
            if 0 <= k_-k <= 1:
                V_.append(v_)
        purity_dict_prune[k] = np.mean(V_)
    if prune==False:
        plt.figure()
        plt.scatter(list(purity_dict.keys()), list(purity_dict.values()), s=3) 
        plt.title('HCP:'+area_name)
    if prune==True:
        plt.figure()
        plt.scatter(list(purity_dict_prune.keys()), list(purity_dict_prune.values()), s=3)
        plt.title('HCP:'+area_name)

Purity_hcp('face', prune=True)
Purity_hcp('place', prune=True)
Purity_hcp('body', prune=True)
Purity_hcp('object', prune=True)



### Hausdorff distance
def Hausdorff_som(A, B):
    A_posision = list(zip(np.where(A!=0)[0], np.where(A!=0)[1]))
    B_posision = list(zip(np.where(B!=0)[0], np.where(B!=0)[1]))
    # A->B
    A2B_distance = []
    for x,y in tqdm(A_posision):
        a2b_distance = []
        for i,j in B_posision:
            a2b_distance.append(np.sqrt((x-i)**2+(y-j)**2))
        A2B_distance.append(min(a2b_distance))
    A2B_distance = max(A2B_distance)
    return A2B_distance

# SOM
d_som_face_place = Hausdorff_som(face_mask, place_mask)
d_som_face_limb = Hausdorff_som(face_mask, limb_mask)
d_som_face_object = Hausdorff_som(face_mask, object_mask)
d_som_place_face = Hausdorff_som(place_mask, face_mask)
d_som_place_limb = Hausdorff_som(place_mask, limb_mask)
d_som_place_object = Hausdorff_som(place_mask, object_mask)
d_som_limb_face = Hausdorff_som(limb_mask, face_mask)
d_som_limb_place = Hausdorff_som(limb_mask, place_mask)
d_som_limb_object = Hausdorff_som(limb_mask, object_mask)
d_som_object_face = Hausdorff_som(object_mask, face_mask)
d_som_object_place = Hausdorff_som(object_mask, place_mask)
d_som_object_limb = Hausdorff_som(object_mask, limb_mask)


# HCP
def Hausdorff_hcp(A_pos, B_pos):
    # A->B
    A2B_distance = []
    for x,y in tqdm(A_pos):
        a2b_distance = []
        for i,j in B_pos:
            a2b_distance.append(np.sqrt((x-i)**2+(y-j)**2))
        A2B_distance.append(min(a2b_distance))
    A2B_distance = max(A2B_distance)
    return A2B_distance

def get_L_hcp_space_mask(hcp_index, threshold, plot=False):
    # geometry information
    geometry = nib.load('/nfs/a2/HCP_S1200/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * L_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
    # make dict
    list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
    CORTEX_Left = list_of_block[0]
    Dict_hcp_to_32kL = dict()
    for vertex in range(CORTEX_Left.index_count):
        Dict_hcp_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(L_hcp_vtc[:CORTEX_Left.index_count]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
        hcp_32K.append(hcp_data[i])
    if plot==False:
        pass
    if plot==True:
        # plot
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(geometry[vtc_32K][:,0], geometry[vtc_32K][:,1], marker='.',
                    c=hcp_32K, cmap=plt.cm.jet)
    class_index = np.where(np.array(hcp_32K)==1)[0].tolist()
    position = []
    for i in class_index:
        position.append(geometry[vtc_32K[i]][[0,1]].tolist())
    position = np.array(position)
    value = hcp_32K
    return position, value

def get_R_hcp_space_mask(hcp_index, threshold, plot=False):
    # geometry information
    geometry = nib.load('/nfs/a2/HCP_S1200/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * R_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
    # make dict
    list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
    CORTEX_Right = list_of_block[1]
    Dict_hcp_to_32kL = dict()
    for vertex in range(CORTEX_Right.index_count):
        Dict_hcp_to_32kL[vertex] = CORTEX_Right.vertex_indices[vertex]
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
        hcp_32K.append(hcp_data[i+29696])
    if plot==False:
        pass
    if plot==True:
        # plot
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(geometry[vtc_32K][:,0], geometry[vtc_32K][:,1], marker='.',
                    c=hcp_32K, cmap=plt.cm.jet)
    class_index = np.where(np.array(hcp_32K)==1)[0].tolist()
    position = []
    for i in class_index:
        position.append(geometry[vtc_32K[i]][[0,1]].tolist())
    position = np.array(position)
    value = hcp_32K
    return position, value

threshold = 0.5
# Left
hcp_face,_ = get_L_hcp_space_mask(19,threshold)
hcp_place,_ = get_L_hcp_space_mask(20,threshold)
hcp_limb,_ = get_L_hcp_space_mask(18,threshold)
hcp_object,_ = get_L_hcp_space_mask(21,threshold)

# Right
hcp_face,_ = get_R_hcp_space_mask(19,threshold)
hcp_place,_ = get_R_hcp_space_mask(20,threshold)
hcp_limb,_ = get_R_hcp_space_mask(18,threshold)
hcp_object,_ = get_R_hcp_space_mask(21,threshold)

d_hcp_face_place = Hausdorff_hcp(hcp_face, hcp_place)
d_hcp_face_limb = Hausdorff_hcp(hcp_face, hcp_limb)
d_hcp_face_object = Hausdorff_hcp(hcp_face, hcp_object)
d_hcp_place_face = Hausdorff_hcp(hcp_place, hcp_face)
d_hcp_place_limb = Hausdorff_hcp(hcp_place, hcp_limb)
d_hcp_place_object = Hausdorff_hcp(hcp_place, hcp_object)
d_hcp_limb_face = Hausdorff_hcp(hcp_limb, hcp_face)
d_hcp_limb_place = Hausdorff_hcp(hcp_limb, hcp_place)
d_hcp_limb_object = Hausdorff_hcp(hcp_limb, hcp_object)
d_hcp_object_face = Hausdorff_hcp(hcp_object, hcp_face)
d_hcp_object_place = Hausdorff_hcp(hcp_object, hcp_place)
d_hcp_object_limb = Hausdorff_hcp(hcp_object, hcp_limb)


plt.figure()
plt.plot([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
          d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
          d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb])
plt.plot([d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
          d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
          d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
   
r_hausdorff = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                   d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                   d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                  [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                   d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                   d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
print('r_hausdorff:', r_hausdorff[0,1])
 


### Dice 
def Dice_som():
    Mask = [face_mask, place_mask, limb_mask, object_mask]
    Dice = []
    for i in range(4):
        for j in range(4):
            if i!=j:
                temp = np.array(Mask[i])+np.array(Mask[j])
                overlap_number = np.where(temp==2)[0].shape[0]
                A_number = np.where(np.array(Mask[i])==1)[0].shape[0]
                B_number = np.where(np.array(Mask[j])==1)[0].shape[0]
                AB_avg_number = (A_number+B_number)/2
                dice = overlap_number/AB_avg_number
                Dice.append(dice)
    return Dice

dice_som = Dice_som()
plt.figure()
plt.plot(dice_som)
                

def Dice_hcp_left():
    threshold_cohend = 0.5
    position, face_mask = get_L_hcp_space_mask(19, threshold_cohend, plot=False)
    position, place_mask = get_L_hcp_space_mask(20, threshold_cohend, plot=False)
    position, limb_mask = get_L_hcp_space_mask(18, threshold_cohend, plot=False)
    position, object_mask = get_L_hcp_space_mask(21, threshold_cohend, plot=False)
    Mask = [face_mask, place_mask, limb_mask, object_mask]
    Dice = []
    for i in range(4):
        for j in range(4):
            if i!=j:
                temp = np.array(Mask[i])+np.array(Mask[j])
                overlap_number = np.where(temp==2)[0].shape[0]
                A_number = np.where(np.array(Mask[i])==1)[0].shape[0]
                B_number = np.where(np.array(Mask[j])==1)[0].shape[0]
                AB_avg_number = (A_number+B_number)/2
                dice = overlap_number/AB_avg_number
                Dice.append(dice)
    return Dice

def Dice_hcp_right():
    threshold_cohend = 0.5
    position, face_mask = get_R_hcp_space_mask(19, threshold_cohend, plot=False)
    position, place_mask = get_R_hcp_space_mask(20, threshold_cohend, plot=False)
    position, limb_mask = get_R_hcp_space_mask(18, threshold_cohend, plot=False)
    position, object_mask = get_R_hcp_space_mask(21, threshold_cohend, plot=False)
    Mask = [face_mask, place_mask, limb_mask, object_mask]
    Dice = []
    for i in range(4):
        for j in range(4):
            if i!=j:
                temp = np.array(Mask[i])+np.array(Mask[j])
                overlap_number = np.where(temp==2)[0].shape[0]
                A_number = np.where(np.array(Mask[i])==1)[0].shape[0]
                B_number = np.where(np.array(Mask[j])==1)[0].shape[0]
                AB_avg_number = (A_number+B_number)/2
                dice = overlap_number/AB_avg_number
                Dice.append(dice)
    return Dice

dice_hcp = Dice_hcp_left()
dice_hcp = Dice_hcp_right()
plt.figure()
plt.plot(dice_hcp)

r_dice = np.corrcoef(dice_som,dice_hcp)
print('r_dice:', r_dice)

print('composite indicators:',
      (r_hausdorff[0,1]+r_dice[0,1])/2)




"""Bayes optimization params"""
###############################################################################
###############################################################################
def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))

def none_decay(scalar, t, max_iter):
    return scalar

def r_hausdorff_indicator(sig, lr):
    som = BrainSOM.VTCSOM(64, 64, 1000, sigma=sig, learning_rate=lr, 
                          sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                          neighborhood_function='gaussian', random_seed=0)
    Data = np.load('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/Data.npy')
    som.pca_weights_init(Data)
    q_error, t_error = som.Train(Data, [0,50000], step_len=100000, verbose=False)
    
    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet.eval()
    model = alexnet
    model_truncate = NET_fc(model, selected_layer=2)
        
    Response_face = Functional_map('face', som)
    Response_place = Functional_map('place', som)
    Response_body = Functional_map('body', som)
    Response_object = Functional_map('object', som)
    
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
    
    t_map, p_map = stats.ttest_1samp(Response_place, Contrast_respense[1])
    place_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response_place[:,i,j], Contrast_respense[1][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
                place_mask[i,j] = 1
    
    t_map, p_map = stats.ttest_1samp(Response_body, Contrast_respense[2])
    limb_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response_body[:,i,j], Contrast_respense[2][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
                limb_mask[i,j] = 1
    
    t_map, p_map = stats.ttest_1samp(Response_object, Contrast_respense[3])
    object_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response_object[:,i,j], Contrast_respense[3][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
                object_mask[i,j] = 1
    
    ### Hausdorff
    ## SOM hausdorff
    d_som_face_place = Hausdorff_som(face_mask, place_mask)
    d_som_face_limb = Hausdorff_som(face_mask, limb_mask)
    d_som_face_object = Hausdorff_som(face_mask, object_mask)
    d_som_place_face = Hausdorff_som(place_mask, face_mask)
    d_som_place_limb = Hausdorff_som(place_mask, limb_mask)
    d_som_place_object = Hausdorff_som(place_mask, object_mask)
    d_som_limb_face = Hausdorff_som(limb_mask, face_mask)
    d_som_limb_place = Hausdorff_som(limb_mask, place_mask)
    d_som_limb_object = Hausdorff_som(limb_mask, object_mask)
    d_som_object_face = Hausdorff_som(object_mask, face_mask)
    d_som_object_place = Hausdorff_som(object_mask, place_mask)
    d_som_object_limb = Hausdorff_som(object_mask, limb_mask)
    
    ## hcp hausdorff
    threshold = 0.5
    # Left
    hcp_face,_ = get_L_hcp_space_mask(19,threshold)
    hcp_place,_ = get_L_hcp_space_mask(20,threshold)
    hcp_limb,_ = get_L_hcp_space_mask(18,threshold)
    hcp_object,_ = get_L_hcp_space_mask(21,threshold)
    
    d_hcp_face_place = Hausdorff_hcp(hcp_face, hcp_place)
    d_hcp_face_limb = Hausdorff_hcp(hcp_face, hcp_limb)
    d_hcp_face_object = Hausdorff_hcp(hcp_face, hcp_object)
    d_hcp_place_face = Hausdorff_hcp(hcp_place, hcp_face)
    d_hcp_place_limb = Hausdorff_hcp(hcp_place, hcp_limb)
    d_hcp_place_object = Hausdorff_hcp(hcp_place, hcp_object)
    d_hcp_limb_face = Hausdorff_hcp(hcp_limb, hcp_face)
    d_hcp_limb_place = Hausdorff_hcp(hcp_limb, hcp_place)
    d_hcp_limb_object = Hausdorff_hcp(hcp_limb, hcp_object)
    d_hcp_object_face = Hausdorff_hcp(hcp_object, hcp_face)
    d_hcp_object_place = Hausdorff_hcp(hcp_object, hcp_place)
    d_hcp_object_limb = Hausdorff_hcp(hcp_object, hcp_limb)       
    r_hausdorff_left = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                       d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                       d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                      [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                       d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                       d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    # Right
    hcp_face,_ = get_R_hcp_space_mask(19,threshold)
    hcp_place,_ = get_R_hcp_space_mask(20,threshold)
    hcp_limb,_ = get_R_hcp_space_mask(18,threshold)
    hcp_object,_ = get_R_hcp_space_mask(21,threshold)
    
    d_hcp_face_place = Hausdorff_hcp(hcp_face, hcp_place)
    d_hcp_face_limb = Hausdorff_hcp(hcp_face, hcp_limb)
    d_hcp_face_object = Hausdorff_hcp(hcp_face, hcp_object)
    d_hcp_place_face = Hausdorff_hcp(hcp_place, hcp_face)
    d_hcp_place_limb = Hausdorff_hcp(hcp_place, hcp_limb)
    d_hcp_place_object = Hausdorff_hcp(hcp_place, hcp_object)
    d_hcp_limb_face = Hausdorff_hcp(hcp_limb, hcp_face)
    d_hcp_limb_place = Hausdorff_hcp(hcp_limb, hcp_place)
    d_hcp_limb_object = Hausdorff_hcp(hcp_limb, hcp_object)
    d_hcp_object_face = Hausdorff_hcp(hcp_object, hcp_face)
    d_hcp_object_place = Hausdorff_hcp(hcp_object, hcp_place)
    d_hcp_object_limb = Hausdorff_hcp(hcp_object, hcp_limb)       
    r_hausdorff_right = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                       d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                       d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                      [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                       d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                       d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    r_hausdorff = max(r_hausdorff_left[0,1], r_hausdorff_right[0,1])  
    return r_hausdorff    

rf_bo = BayesianOptimization(
        r_hausdorff_indicator,
        {'sig': (0.1, 5),
        'lr': (0.1, 2)}) 

rf_bo.maximize(n_iter=20)

print(rf_bo.max)

    
    
    
    
"""Functional connection"""
###############################################################################
###############################################################################
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
M = np.where((M>=np.percentile(M,99.9)) | (M<=np.percentile(M,0.1)), M, 0)
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




"""Soft indicators"""
###############################################################################
###############################################################################
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


