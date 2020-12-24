#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial.distance import directed_hausdorff
import scipy.stats as stats
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from dipy.viz import regtools
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
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()


threshold_cohend = 0.5
HCP_data = nib.load('/home/dell/TDCNN/HCP/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask = nib.load('/home/dell/TDCNN/HCP/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==135)|(mask==136)|(mask==138)|(mask==154)|(mask==163)|(mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==135+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
hcp_vtc = np.zeros(91282)
hcp_vtc[vtc_mask] = 1
R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==135)|(mask==136)|(mask==138)|(mask==154)|(mask==163))[0]
R_hcp_vtc = np.zeros(91282)
R_hcp_vtc[R_vtc_mask] = 1
L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==135+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
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




"""Symmetric Diffeomorphic Registration"""
###############################################################################
###############################################################################
def get_Lvtc_position(plot=False):
    # geometry information
    geometry = nib.load('/nfs/a2/HCP_S1200/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('/home/dell/TDCNN/HCP/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==135+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
    L_hcp_vtc = np.zeros(91282)
    L_hcp_vtc[L_vtc_mask] = 1
    # make dict
    list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
    CORTEX_Left = list_of_block[0]
    Dict_hcp_to_32kL = dict()
    for vertex in range(CORTEX_Left.index_count):
        Dict_hcp_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
    # hcp mapping to 32K
    vtc_32K = []
    for i in np.where(L_hcp_vtc[:CORTEX_Left.index_count]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
    position = geometry[vtc_32K][:,[0,1]]
    if plot==False:
        pass
    if plot==True:
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(position[:,0], position[:,1], marker='s')
    return position

def get_Rvtc_position(plot=False):
    # geometry information
    geometry = nib.load('/nfs/a2/HCP_S1200/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('/home/dell/TDCNN/HCP/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==135)|(mask==136)|(mask==138)|(mask==154)|(mask==163))[0]
    R_hcp_vtc = np.zeros(91282)
    R_hcp_vtc[R_vtc_mask] = 1
    # make dict
    list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
    CORTEX_Right = list_of_block[1]
    Dict_hcp_to_32kL = dict()
    for vertex in range(CORTEX_Right.index_count):
        Dict_hcp_to_32kL[vertex] = CORTEX_Right.vertex_indices[vertex]
    # hcp mapping to 32K
    vtc_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
    position = geometry[vtc_32K][:,[0,1]]
    if plot==False:
        pass
    if plot==True:
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(position[:,0], position[:,1], marker='s')
    return position

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

def fill_hcp_map(position):
    round_position = []
    for v in range(position.shape[0]):
        round_position.append([np.floor(position[v,:][0]), np.floor(position[v,:][1])])
        round_position.append([np.floor(position[v,:][0]), np.ceil(position[v,:][1])])
        round_position.append([np.ceil(position[v,:][0]), np.floor(position[v,:][1])])
        round_position.append([np.ceil(position[v,:][0]), np.ceil(position[v,:][1])])
    round_position = np.array(round_position)
    return round_position
 
def make_moving_map(position, hemisphere):
    '''
    hemisphere is 'left' or 'right'
    '''
    round_position = fill_hcp_map(position)
    if hemisphere=='left':
        vtc_round_position = fill_hcp_map(get_Lvtc_position(plot=False))
    if hemisphere=='right':
        vtc_round_position = fill_hcp_map(get_Rvtc_position(plot=False))    
    round_position[:,0] -= vtc_round_position[:,0].min()-50
    round_position[:,1] -= vtc_round_position[:,1].min()-50
    round_position = np.int0(round_position)
    moving = np.zeros((220,220))
    for pos in round_position:
        moving[pos[0],pos[1]] = 1
    moving_copy = copy.deepcopy(moving)
    it = np.nditer(moving, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
            for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                if moving_copy[it.multi_index]==1:
                    moving[ii,jj] = 1
        it.iternext()
    return moving
    
def Make_mapping_vtc2sheet(position, hemisphere):
    moving = make_moving_map(position, hemisphere)
    static = np.zeros((220,220))  
    static[10:210,10:210] = 1
    regtools.overlay_images(static, moving, 'Static', 'Overlay', 'Moving')    
    dim = static.ndim
    metric = SSDMetric(dim)    
    level_iters = [500, 200, 100, 50, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving)
    regtools.plot_2d_diffeomorphic_map(mapping)   
    warped_moving = mapping.transform(moving, 'linear')
    regtools.overlay_images(static, warped_moving, 'Static', 'Overlay', 'Warped moving')
    return mapping

def Mapping_area2sheet(mapping, area_position, hemisphere):
    moving = make_moving_map(area_position, hemisphere)
    static = np.zeros((220,220))  
    static[10:210,10:210] = 1    
    warped_moving = mapping.transform(moving, 'linear')
    warped_moving = warped_moving[10:210,10:210]
#    plt.figure()
#    plt.imshow(warped_moving)
    return warped_moving

mapping = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), hemisphere='left')

threshold = 0.5
hcp_face,_ = get_L_hcp_space_mask(19, threshold)
warped_face = Mapping_area2sheet(mapping, hcp_face, hemisphere='left')
hcp_place,_ = get_L_hcp_space_mask(20, threshold)
warped_place = Mapping_area2sheet(mapping, hcp_place, hemisphere='left')
hcp_limb,_ = get_L_hcp_space_mask(18, threshold)
warped_limb = Mapping_area2sheet(mapping, hcp_limb, hemisphere='left')
hcp_object,_ = get_L_hcp_space_mask(21, threshold)
warped_object = Mapping_area2sheet(mapping, hcp_object, hemisphere='left')




"""Hausdorff Distance"""
###############################################################################
###############################################################################    
def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map(class_name, som, pca_type, pca_index):  
    '''
    pca_type: mypca comes from test_data, pca comes from huangtaicheng
    pca_index: a list of features, such as [0,1,2,3,4]
    '''
    f = os.listdir('/home/dell/TDCNN/HCP/HCP_WM/' + class_name)
    f.remove('.DS_Store')
    if pca_type=='mypca':
        Data = np.load('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/Data.npy')
        pca = PCA()
        pca.fit(Data)
        Response = []
        for index,pic in enumerate(f):
            img = Image.open('/home/dell/TDCNN/HCP/HCP_WM/'+class_name+'/'+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(1/som.activate(pca.transform(output)[:,pca_index]))   
        Response = np.array(Response)
    if pca_type=='pca':
        pca = np.load('/home/dell/TDCNN/pca_imgnet_50w_mod1.pkl', allow_pickle=True)
        Response = []
        for index,pic in enumerate(f):
            img = Image.open('/home/dell/TDCNN/HCP/HCP_WM/'+class_name+'/'+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(1/som.activate(pca.transform(output)[:,pca_index]))   
        Response = np.array(Response)    
    return Response

def som_mask(som, Response, Contrast_respense, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[0])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[0][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask

def get_mask_position(mask):
    return list(zip(np.where(mask!=0)[0], np.where(mask!=0)[1]))
  
def hausdorff_Left(weights_dir, pca_type, pca_index):
    som = BrainSOM.VTCSOM(200, 200, len(pca_index), sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(weights_dir)
        
    Response_face = Functional_map('face', som, pca_type, pca_index)
    Response_place = Functional_map('place', som, pca_type, pca_index)
    Response_body = Functional_map('body', som, pca_type, pca_index)
    Response_object = Functional_map('object', som, pca_type, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, threshold_cohend)
    
    ### Hausdorff
    ## SOM hausdorff 
    d_som_face_place = directed_hausdorff(get_mask_position(face_mask), get_mask_position(place_mask))[0]
    d_som_face_limb = directed_hausdorff(get_mask_position(face_mask), get_mask_position(limb_mask))[0]
    d_som_face_object = directed_hausdorff(get_mask_position(face_mask), get_mask_position(object_mask))[0]
    d_som_place_face = directed_hausdorff(get_mask_position(place_mask), get_mask_position(face_mask))[0]
    d_som_place_limb = directed_hausdorff(get_mask_position(place_mask), get_mask_position(limb_mask))[0]
    d_som_place_object = directed_hausdorff(get_mask_position(place_mask), get_mask_position(object_mask))[0]
    d_som_limb_face = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(face_mask))[0]
    d_som_limb_place = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(place_mask))[0]
    d_som_limb_object = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(object_mask))[0]
    d_som_object_face = directed_hausdorff(get_mask_position(object_mask), get_mask_position(face_mask))[0]
    d_som_object_place = directed_hausdorff(get_mask_position(object_mask), get_mask_position(place_mask))[0]
    d_som_object_limb = directed_hausdorff(get_mask_position(object_mask), get_mask_position(limb_mask))[0]
    
    ## hcp hausdorff (Left)
    threshold = 0.5
    hcp_face,_ = get_L_hcp_space_mask(19,threshold)
    hcp_place,_ = get_L_hcp_space_mask(20,threshold)
    hcp_limb,_ = get_L_hcp_space_mask(18,threshold)
    hcp_object,_ = get_L_hcp_space_mask(21,threshold)
    
    d_hcp_face_place = directed_hausdorff(hcp_face, hcp_place)[0]
    d_hcp_face_limb = directed_hausdorff(hcp_face, hcp_limb)[0]
    d_hcp_face_object = directed_hausdorff(hcp_face, hcp_object)[0]
    d_hcp_place_face = directed_hausdorff(hcp_place, hcp_face)[0]
    d_hcp_place_limb = directed_hausdorff(hcp_place, hcp_limb)[0]
    d_hcp_place_object = directed_hausdorff(hcp_place, hcp_object)[0]
    d_hcp_limb_face = directed_hausdorff(hcp_limb, hcp_face)[0]
    d_hcp_limb_place = directed_hausdorff(hcp_limb, hcp_place)[0]
    d_hcp_limb_object = directed_hausdorff(hcp_limb, hcp_object)[0]
    d_hcp_object_face = directed_hausdorff(hcp_object, hcp_face)[0]
    d_hcp_object_place = directed_hausdorff(hcp_object, hcp_place)[0]
    d_hcp_object_limb = directed_hausdorff(hcp_object, hcp_limb)[0]       
    r_hausdorff_left = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                                    d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                                    d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                                   [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                                    d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                                    d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    r_hausdorff = r_hausdorff_left[0,1] 
    return r_hausdorff   


def hausdorff_Right(weights_dir, pca_type, pca_index):
    som = BrainSOM.VTCSOM(200, 200, len(pca_index), sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(weights_dir)
        
    Response_face = Functional_map('face', som, pca_type, pca_index)
    Response_place = Functional_map('place', som, pca_type, pca_index)
    Response_body = Functional_map('body', som, pca_type, pca_index)
    Response_object = Functional_map('object', som, pca_type, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, threshold_cohend)
    
    ### Hausdorff
    ## SOM hausdorff
    d_som_face_place = directed_hausdorff(get_mask_position(face_mask), get_mask_position(place_mask))[0]
    d_som_face_limb = directed_hausdorff(get_mask_position(face_mask), get_mask_position(limb_mask))[0]
    d_som_face_object = directed_hausdorff(get_mask_position(face_mask), get_mask_position(object_mask))[0]
    d_som_place_face = directed_hausdorff(get_mask_position(place_mask), get_mask_position(face_mask))[0]
    d_som_place_limb = directed_hausdorff(get_mask_position(place_mask), get_mask_position(limb_mask))[0]
    d_som_place_object = directed_hausdorff(get_mask_position(place_mask), get_mask_position(object_mask))[0]
    d_som_limb_face = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(face_mask))[0]
    d_som_limb_place = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(place_mask))[0]
    d_som_limb_object = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(object_mask))[0]
    d_som_object_face = directed_hausdorff(get_mask_position(object_mask), get_mask_position(face_mask))[0]
    d_som_object_place = directed_hausdorff(get_mask_position(object_mask), get_mask_position(place_mask))[0]
    d_som_object_limb = directed_hausdorff(get_mask_position(object_mask), get_mask_position(limb_mask))[0]   
    
    ## hcp hausdorff (Right)
    threshold = 0.5
    hcp_face,_ = get_R_hcp_space_mask(19,threshold)
    hcp_place,_ = get_R_hcp_space_mask(20,threshold)
    hcp_limb,_ = get_R_hcp_space_mask(18,threshold)
    hcp_object,_ = get_R_hcp_space_mask(21,threshold)
    
    d_hcp_face_place = directed_hausdorff(hcp_face, hcp_place)[0]
    d_hcp_face_limb = directed_hausdorff(hcp_face, hcp_limb)[0]
    d_hcp_face_object = directed_hausdorff(hcp_face, hcp_object)[0]
    d_hcp_place_face = directed_hausdorff(hcp_place, hcp_face)[0]
    d_hcp_place_limb = directed_hausdorff(hcp_place, hcp_limb)[0]
    d_hcp_place_object = directed_hausdorff(hcp_place, hcp_object)[0]
    d_hcp_limb_face = directed_hausdorff(hcp_limb, hcp_face)[0]
    d_hcp_limb_place = directed_hausdorff(hcp_limb, hcp_place)[0]
    d_hcp_limb_object = directed_hausdorff(hcp_limb, hcp_object)[0]
    d_hcp_object_face = directed_hausdorff(hcp_object, hcp_face)[0]
    d_hcp_object_place = directed_hausdorff(hcp_object, hcp_place)[0]
    d_hcp_object_limb = directed_hausdorff(hcp_object, hcp_limb)[0]       
    r_hausdorff_right = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                                     d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                                     d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                                    [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                                     d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                                     d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    r_hausdorff = r_hausdorff_right[0,1] 
    return r_hausdorff    


def run_hausdorff_in_varied_sigma(weights_dir, out_file, pca_type, pca_index):
    f = os.listdir(weights_dir)
    for i in f:
        if i[:3]!='som':
            f.remove(i)
    f.sort()
    with open(out_file, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(['Sigma', 'r_hausdorff_left', 'r_hausdorff_right'])
        for w_file in tqdm(f):
            w_dir = weights_dir+w_file
            r_left = hausdorff_Left(w_dir, pca_type, pca_index)
            r_right = hausdorff_Right(w_dir, pca_type, pca_index)
            csv_write.writerow([w_file[10:13], r_left, r_right])
            
def plot_r_hausdorff(csv_file, medfilt_range):
    plt.figure()
    file = pd.read_csv(csv_file)
    r_hausdorff_left = file['r_hausdorff_left'].tolist()
    r_hausdorff_right = file['r_hausdorff_right'].tolist()
    plt.plot(signal.medfilt(r_hausdorff_left, medfilt_range))
    plt.plot(signal.medfilt(r_hausdorff_right, medfilt_range))



run_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/Sig_hausdorff.csv', 
                              'mypca',
                              [0,1,2,3])
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/Sig_hausdorff.csv', 
                 medfilt_range=3)

run_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/Sig_hausdorff.csv', 
                              'mypca',
                              [0,1,2,3,4])
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/Sig_hausdorff.csv', 
                 medfilt_range=3)

run_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/Sig_hausdorff.csv', 
                              'pca',
                              [0,1,2,3])    
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/Sig_hausdorff.csv', 
                 medfilt_range=3)  
                       
run_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/Sig_hausdorff.csv', 
                              'pca',
                              [0,1,2,3,4])    
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/Sig_hausdorff.csv', 
                 medfilt_range=3)                         





"""Symmetric Diffeomorphic Registration + Hausdorff Distance"""
###############################################################################
###############################################################################
def SDR_hausdorff_Left(weights_dir, mapping, pca_type, pca_index, hemisphere):
    som = BrainSOM.VTCSOM(200, 200, 5, sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(weights_dir)
        
    Response_face = Functional_map('face', som, pca_type, pca_index)
    Response_place = Functional_map('place', som, pca_type, pca_index)
    Response_body = Functional_map('body', som, pca_type, pca_index)
    Response_object = Functional_map('object', som, pca_type, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, threshold_cohend)
    
    ### Hausdorff
    ## SOM hausdorff 
    d_som_face_place = directed_hausdorff(get_mask_position(face_mask), get_mask_position(place_mask))[0]
    d_som_face_limb = directed_hausdorff(get_mask_position(face_mask), get_mask_position(limb_mask))[0]
    d_som_face_object = directed_hausdorff(get_mask_position(face_mask), get_mask_position(object_mask))[0]
    d_som_place_face = directed_hausdorff(get_mask_position(place_mask), get_mask_position(face_mask))[0]
    d_som_place_limb = directed_hausdorff(get_mask_position(place_mask), get_mask_position(limb_mask))[0]
    d_som_place_object = directed_hausdorff(get_mask_position(place_mask), get_mask_position(object_mask))[0]
    d_som_limb_face = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(face_mask))[0]
    d_som_limb_place = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(place_mask))[0]
    d_som_limb_object = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(object_mask))[0]
    d_som_object_face = directed_hausdorff(get_mask_position(object_mask), get_mask_position(face_mask))[0]
    d_som_object_place = directed_hausdorff(get_mask_position(object_mask), get_mask_position(place_mask))[0]
    d_som_object_limb = directed_hausdorff(get_mask_position(object_mask), get_mask_position(limb_mask))[0]
    
    ## hcp hausdorff (Left)
    threshold = 0.5
    hcp_face,_ = get_L_hcp_space_mask(19, threshold)
    warped_face = Mapping_area2sheet(mapping, hcp_face, hemisphere='left')
    hcp_place,_ = get_L_hcp_space_mask(20, threshold)
    warped_place = Mapping_area2sheet(mapping, hcp_place, hemisphere='left')
    hcp_limb,_ = get_L_hcp_space_mask(18, threshold)
    warped_limb = Mapping_area2sheet(mapping, hcp_limb, hemisphere='left')
    hcp_object,_ = get_L_hcp_space_mask(21, threshold)
    warped_object = Mapping_area2sheet(mapping, hcp_object, hemisphere='left')
    hcp_face = get_mask_position(warped_face)
    hcp_place = get_mask_position(warped_place)
    hcp_limb = get_mask_position(warped_limb)
    hcp_object = get_mask_position(warped_object)
    
    d_hcp_face_place = directed_hausdorff(hcp_face, hcp_place)[0]
    d_hcp_face_limb = directed_hausdorff(hcp_face, hcp_limb)[0]
    d_hcp_face_object = directed_hausdorff(hcp_face, hcp_object)[0]
    d_hcp_place_face = directed_hausdorff(hcp_place, hcp_face)[0]
    d_hcp_place_limb = directed_hausdorff(hcp_place, hcp_limb)[0]
    d_hcp_place_object = directed_hausdorff(hcp_place, hcp_object)[0]
    d_hcp_limb_face = directed_hausdorff(hcp_limb, hcp_face)[0]
    d_hcp_limb_place = directed_hausdorff(hcp_limb, hcp_place)[0]
    d_hcp_limb_object = directed_hausdorff(hcp_limb, hcp_object)[0]
    d_hcp_object_face = directed_hausdorff(hcp_object, hcp_face)[0]
    d_hcp_object_place = directed_hausdorff(hcp_object, hcp_place)[0]
    d_hcp_object_limb = directed_hausdorff(hcp_object, hcp_limb)[0]       
    r_hausdorff_left = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                                    d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                                    d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                                   [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                                    d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                                    d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    r_hausdorff = r_hausdorff_left[0,1] 
    return r_hausdorff   

def SDR_hausdorff_Right(weights_dir, mapping, pca_type, pca_index, hemisphere):
    som = BrainSOM.VTCSOM(200, 200, 5, sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(weights_dir)
        
    Response_face = Functional_map('face', som, pca_type, pca_index)
    Response_place = Functional_map('place', som, pca_type, pca_index)
    Response_body = Functional_map('body', som, pca_type, pca_index)
    Response_object = Functional_map('object', som, pca_type, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, threshold_cohend)
    
    ### Hausdorff
    ## SOM hausdorff 
    d_som_face_place = directed_hausdorff(get_mask_position(face_mask), get_mask_position(place_mask))[0]
    d_som_face_limb = directed_hausdorff(get_mask_position(face_mask), get_mask_position(limb_mask))[0]
    d_som_face_object = directed_hausdorff(get_mask_position(face_mask), get_mask_position(object_mask))[0]
    d_som_place_face = directed_hausdorff(get_mask_position(place_mask), get_mask_position(face_mask))[0]
    d_som_place_limb = directed_hausdorff(get_mask_position(place_mask), get_mask_position(limb_mask))[0]
    d_som_place_object = directed_hausdorff(get_mask_position(place_mask), get_mask_position(object_mask))[0]
    d_som_limb_face = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(face_mask))[0]
    d_som_limb_place = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(place_mask))[0]
    d_som_limb_object = directed_hausdorff(get_mask_position(limb_mask), get_mask_position(object_mask))[0]
    d_som_object_face = directed_hausdorff(get_mask_position(object_mask), get_mask_position(face_mask))[0]
    d_som_object_place = directed_hausdorff(get_mask_position(object_mask), get_mask_position(place_mask))[0]
    d_som_object_limb = directed_hausdorff(get_mask_position(object_mask), get_mask_position(limb_mask))[0]
    
    ## hcp hausdorff (Right)
    threshold = 0.5
    hcp_face,_ = get_R_hcp_space_mask(19, threshold)
    warped_face = Mapping_area2sheet(mapping, hcp_face, hemisphere='right')
    hcp_place,_ = get_R_hcp_space_mask(20, threshold)
    warped_place = Mapping_area2sheet(mapping, hcp_place, hemisphere='right')
    hcp_limb,_ = get_R_hcp_space_mask(18, threshold)
    warped_limb = Mapping_area2sheet(mapping, hcp_limb, hemisphere='right')
    hcp_object,_ = get_R_hcp_space_mask(21, threshold)
    warped_object = Mapping_area2sheet(mapping, hcp_object, hemisphere='right')
    hcp_face = get_mask_position(warped_face)
    hcp_place = get_mask_position(warped_place)
    hcp_limb = get_mask_position(warped_limb)
    hcp_object = get_mask_position(warped_object)
    
    d_hcp_face_place = directed_hausdorff(hcp_face, hcp_place)[0]
    d_hcp_face_limb = directed_hausdorff(hcp_face, hcp_limb)[0]
    d_hcp_face_object = directed_hausdorff(hcp_face, hcp_object)[0]
    d_hcp_place_face = directed_hausdorff(hcp_place, hcp_face)[0]
    d_hcp_place_limb = directed_hausdorff(hcp_place, hcp_limb)[0]
    d_hcp_place_object = directed_hausdorff(hcp_place, hcp_object)[0]
    d_hcp_limb_face = directed_hausdorff(hcp_limb, hcp_face)[0]
    d_hcp_limb_place = directed_hausdorff(hcp_limb, hcp_place)[0]
    d_hcp_limb_object = directed_hausdorff(hcp_limb, hcp_object)[0]
    d_hcp_object_face = directed_hausdorff(hcp_object, hcp_face)[0]
    d_hcp_object_place = directed_hausdorff(hcp_object, hcp_place)[0]
    d_hcp_object_limb = directed_hausdorff(hcp_object, hcp_limb)[0]       
    r_hausdorff_right = np.corrcoef([d_som_face_place, d_som_face_limb, d_som_face_object, d_som_place_face,
                                    d_som_place_limb, d_som_place_object, d_som_limb_face, d_som_limb_place,
                                    d_som_limb_object, d_som_object_face, d_som_object_place, d_som_object_limb],
                                   [d_hcp_face_place, d_hcp_face_limb, d_hcp_face_object, d_hcp_place_face,
                                    d_hcp_place_limb, d_hcp_place_object, d_hcp_limb_face, d_hcp_limb_place,
                                    d_hcp_limb_object, d_hcp_object_face, d_hcp_object_place, d_hcp_object_limb])
    r_hausdorff = r_hausdorff_right[0,1] 
    return r_hausdorff   


def run_SDR_hausdorff_in_varied_sigma(weights_dir, out_file, mapping_Left, mapping_Right, pca_type, pca_index):
    f = os.listdir(weights_dir)
    for i in f:
        if i[:3]!='som':
            f.remove(i)
    f.sort()
    with open(out_file, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(['Sigma', 'r_hausdorff_left', 'r_hausdorff_right'])
        for w_file in tqdm(f):
            w_dir = weights_dir+w_file
            r_left = SDR_hausdorff_Left(w_dir, mapping_Left, pca_type, pca_index, hemisphere='left')
            r_right = SDR_hausdorff_Right(w_dir, mapping_Right, pca_type, pca_index, hemisphere='right')
            csv_write.writerow([w_file[10:13], r_left, r_right])

def plot_r_hausdorff(csv_file, medfilt_range):
    plt.figure()
    file = pd.read_csv(csv_file)
    r_hausdorff_left = file['r_hausdorff_left'].tolist()
    r_hausdorff_right = file['r_hausdorff_right'].tolist()
    plt.plot(signal.medfilt(r_hausdorff_left, medfilt_range))
    plt.plot(signal.medfilt(r_hausdorff_right, medfilt_range))



mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), hemisphere='left')
mapping_Right = Make_mapping_vtc2sheet(get_Rvtc_position(plot=False), hemisphere='right')
run_SDR_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                              mapping_Left, mapping_Right, 'mypca', [0,1,2,3])
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca4_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                 medfilt_range=3)

run_SDR_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                              mapping_Left, mapping_Right, 'mypca', [0,1,2,3,4])
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_mypca5_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                 medfilt_range=3)

run_SDR_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                              mapping_Left, mapping_Right, 'pca', [0,1,2,3])    
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca4_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                 medfilt_range=3)  
                       
run_SDR_hausdorff_in_varied_sigma('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/', 
                              '/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                              mapping_Left, mapping_Right, 'pca', [0,1,2,3,4])    
plot_r_hausdorff('/home/dell/TDCNN/Results/Alexnet_fc8_SOM/SOM(200x200)_pca5_Sigma_200000step/Sig_SDR_hausdorff.csv', 
                 medfilt_range=3)           



