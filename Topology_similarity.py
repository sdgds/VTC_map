#!\usr\bin\env python3
# -*- coding: utf-8 -*-
import os
import cv2
import copy
import csv
from tqdm import tqdm
import numpy as np
from skimage import transform
from scipy.stats import zscore
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
sys.path.append(r'D:\TDCNN\\')
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
HCP_data = nib.load('D:\TDCNN\HCP\HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask = nib.load('D:\TDCNN\HCP\MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
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
# make dict
list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
CORTEX_Left = list_of_block[0]
Dict_hcp_to_32kL = dict()
for vertex in range(CORTEX_Left.index_count):
    Dict_hcp_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
Dict_32kL_to_hcp = {v:k for k,v in Dict_hcp_to_32kL.items()}

list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
CORTEX_Right = list_of_block[1]
Dict_hcp_to_32kR = dict()
for vertex in range(CORTEX_Right.index_count):
    Dict_hcp_to_32kR[vertex] = CORTEX_Right.vertex_indices[vertex]
Dict_32kR_to_hcp = {v:k for k,v in Dict_hcp_to_32kR.items()}
        
def get_Lvtc_position(plot=False):
    # geometry information
    geometry = nib.load('D:\TDCNN\HCP\S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('D:\TDCNN\HCP\MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==135+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
    L_hcp_vtc = np.zeros(91282)
    L_hcp_vtc[L_vtc_mask] = 1
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
    geometry = nib.load('D:\TDCNN\HCP\S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('D:\TDCNN\HCP\MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==135)|(mask==136)|(mask==138)|(mask==154)|(mask==163))[0]
    R_hcp_vtc = np.zeros(91282)
    R_hcp_vtc[R_vtc_mask] = 1
    # hcp mapping to 32K
    vtc_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kR[i])
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
    geometry = nib.load('D:\TDCNN\HCP\S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * L_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
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
    geometry = nib.load('D:\TDCNN\HCP\S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * R_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kR[i])
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
 
def make_moving_map(position, rotation_theta, hemisphere):
    '''
    hemisphere is 'left' or 'right'
    '''
    round_position = fill_hcp_map(position)
    if hemisphere=='left':
        vtc_round_position = fill_hcp_map(get_Lvtc_position(plot=False))
    if hemisphere=='right':
        vtc_round_position = fill_hcp_map(get_Rvtc_position(plot=False))    
    round_position[:,0] -= vtc_round_position[:,0].min()-90
    round_position[:,1] -= vtc_round_position[:,1].min()-70
    round_position = np.int0(round_position)
    moving = np.zeros((250,250))
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
    moving = transform.rotate(moving, rotation_theta)
    return moving
    
def Make_mapping_vtc2sheet(position, rotation_theta, hemisphere):
    moving = make_moving_map(position, rotation_theta, hemisphere)
    static = np.zeros((250,250))  
    static[20:220,20:220] = 1
    # regtools.overlay_images(static, moving, 'Static', 'Overlay', 'Moving')    
    dim = static.ndim
    metric = SSDMetric(dim)    
    level_iters = [500, 200, 100, 50, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving)
    # regtools.plot_2d_diffeomorphic_map(mapping)   
    # warped_moving = mapping.transform(moving, 'linear')
    # regtools.overlay_images(static, warped_moving, 'Static', 'Overlay', 'Warped moving')
    return mapping

def Mapping_area2sheet(mapping, area_position, rotation_theta, hemisphere):
    moving = make_moving_map(area_position, rotation_theta, hemisphere)
    static = np.zeros((250,250))  
    static[20:220,20:220] = 1    
    warped_moving = mapping.transform(moving, 'linear')
    warped_moving = warped_moving[20:220,20:220]
    return warped_moving

def Mapping_som_area_2_vtc(mapping, som_area):
    '''som_area: array (200x200)'''      
    moving = np.zeros((250,250))  
    moving[20:220,20:220] = som_area
    vtc_area = mapping.transform_inverse(moving)    
    return vtc_area

def drive_area_2_vtc_axis(warped_area, inverse_rotation_theta, hemisphere):
    if hemisphere=='left':
        position = get_Lvtc_position(plot=False)
        geometry = nib.load('D:\TDCNN\HCP\S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
        Dict_32k_to_hcp = Dict_32kL_to_hcp
    if hemisphere=='right':
        position = get_Rvtc_position(plot=False)
        geometry = nib.load('D:\TDCNN\HCP\S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
        Dict_32k_to_hcp = Dict_32kR_to_hcp
    warped_area = transform.rotate(warped_area, inverse_rotation_theta)
    vtc_round_position = fill_hcp_map(position)
    Xs = np.float32(np.where(warped_area!=0)[0])
    Ys = np.float32(np.where(warped_area!=0)[1])
    Xs += vtc_round_position[:,0].min()-90
    Ys += vtc_round_position[:,1].min()-70
    units_position = []
    units_index_in_32K = []
    for unit in zip(Xs,Ys):
        temp = np.abs(position-unit)
        t = temp[:,0] + temp[:,1]
        units_position.append(position[t.argmin(),:])
        units_index_in_32K.append(np.where(geometry[:,[0,1]]==position[t.argmin(),:])[0][0])
    units_position = np.array(units_position)  
    units_index_in_32K = np.array(units_index_in_32K)
    units_index_in_hcp = []
    for i in units_index_in_32K:
        units_index_in_hcp.append(Dict_32k_to_hcp[i])
    units_in_hcp = np.zeros(91282)
    if hemisphere=='left':
        units_in_hcp[units_index_in_hcp] = 1
    if hemisphere=='right':
        units_in_hcp[[x+29696 for x in units_index_in_hcp]] = 1
    return units_in_hcp

def save_warped_area_as_gii(units_in_hcp, out_dir):
    """"out_dir: .dtseries.nii"""
    data_nii = nib.load('D:\TDCNN\HCP\seg1_1_Atlas.dtseries.nii')
    img = np.tile(units_in_hcp, (245,1))
    IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
    nib.save(IMG, out_dir)

def Mapping_som_units_2_vtc_units(mapping, som, sigma, rotation_theta, hemisphere, units):
    ''''units: [100:102,100:102]'''
    if hemisphere=='left':
        vtc = make_moving_map(get_Lvtc_position(plot=False), rotation_theta, hemisphere)
    if hemisphere=='right':
        vtc = make_moving_map(get_Rvtc_position(plot=False), rotation_theta, hemisphere)        
    som_units = np.zeros((200,200))
    som_units[units] = 1
    moving = np.zeros((250,250))  
    moving[20:220,20:220] = som_units
    vtc_units = mapping.transform_inverse(moving)    
    plt.figure(figsize=(12,12))
    plt.imshow(som_units);plt.axis('off')
    plt.figure(figsize=(12,12))
    plt.imshow(vtc, alpha=1);plt.axis('off')
    plt.imshow(vtc_units, alpha=0.5, cmap='jet');plt.axis('off')
    return som_units, vtc_units

def Mapping_som_structure_constrain_2_vtc_map(mapping, som, sigma, rotation_theta, hemisphere, plot_number):
    if hemisphere=='left':
        vtc = make_moving_map(get_Lvtc_position(plot=False), rotation_theta, hemisphere)
    if hemisphere=='right':
        vtc = make_moving_map(get_Rvtc_position(plot=False), rotation_theta, hemisphere)        
    x = np.random.choice(np.arange(0,200,1), plot_number)
    y = np.random.choice(np.arange(0,200,1), plot_number)
    point_pair = zip(x,y)
    som_structure = np.zeros((200,200))
    vtc_structure_constrain = np.zeros((250,250))
    for point in point_pair:
        som_structure_constrain = som._gaussian(point,sigma)
        som_structure += som_structure_constrain
        moving = np.zeros((250,250))  
        moving[20:220,20:220] = som_structure_constrain
        vtc_structure_constrain += mapping.transform_inverse(moving)    
    plt.figure(figsize=(12,12))
    plt.imshow(som_structure);plt.axis('off')
    plt.figure(figsize=(12,12))
    plt.imshow(vtc, alpha=1);plt.axis('off')
    plt.imshow(vtc_structure_constrain, alpha=0.5, cmap='jet');plt.axis('off')
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    xx = np.arange(0,250,1)
    yy = -np.arange(-250,0,1)
    X, Y = np.meshgrid(xx, yy)
    surf = ax.plot_surface(X,Y,vtc_structure_constrain, cmap='jet')
    ax.set_zlim3d(0)
    fig.colorbar(surf)
    return som_structure, vtc_structure_constrain
   
def plot_vertex_distance(vertex_dir, hemisphere):
    if hemisphere=='left':
        geometry = nib.load('D:\TDCNN\HCP\S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    if hemisphere=='right':
        geometry = nib.load('D:\TDCNN\HCP\S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    hcp_data = nib.load(vertex_dir).darrays[0].data
    hcp_data = np.where(hcp_data>0)[0]
    plt.figure()
    plt.scatter(geometry[:,0], geometry[:,1], marker='.')
    plt.scatter(geometry[hcp_data][:,0], geometry[hcp_data][:,1], marker='.', cmap=plt.cm.jet)


som = BrainSOM.VTCSOM(200, 200, 4, sigma=5, learning_rate=1, neighborhood_function='gaussian', random_seed=0)
som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_6.2.npy')
Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca_index=[0,1,2,3])
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
face_mask = som_mask(som, Response_face, Contrast_respense, 0, 0.5)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, 0.5)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, 0.5)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, 0.5)
plt.figure(figsize=(10,2))
ax1 = plt.subplot(141)
ax1.imshow(face_mask, cmap='Reds', label='face');ax1.axis('off')
ax2 = plt.subplot(142)
ax2.imshow(place_mask, cmap='Greens', label='place');ax2.axis('off')
ax3 = plt.subplot(143)
ax3.imshow(limb_mask, cmap='Oranges', label='limb');ax3.axis('off')
ax4 = plt.subplot(144)
ax4.imshow(object_mask, cmap='Blues', label='object');ax4.axis('off')
# Left hemisphere
for rotation_theta in np.arange(0,360,10):
    mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), rotation_theta, hemisphere='left')    
    threshold = 0.5
    warped_face = Mapping_som_area_2_vtc(mapping_Left, face_mask)
    warped_place = Mapping_som_area_2_vtc(mapping_Left, place_mask)
    warped_limb = Mapping_som_area_2_vtc(mapping_Left, limb_mask)
    warped_object = Mapping_som_area_2_vtc(mapping_Left, object_mask)
    plt.figure(figsize=(15,3))
    vtc = make_moving_map(get_Lvtc_position(plot=False), rotation_theta, 'left')
    ax1 = plt.subplot(141)
    ax1.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax1.imshow(warped_face, cmap='Reds', alpha=0.5);ax1.axis('off')
    ax1.set_title(str(rotation_theta))
    ax2 = plt.subplot(142)
    ax2.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax2.imshow(warped_place, cmap='Greens', alpha=0.5);ax2.axis('off')
    ax3 = plt.subplot(143)
    ax3.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax3.imshow(warped_limb, cmap='Oranges', alpha=0.5);ax3.axis('off')
    ax4 = plt.subplot(144)
    ax4.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax4.imshow(warped_object, cmap='Blues', alpha=0.5);ax4.axis('off')
    # save
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_face, -rotation_theta, 'left'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_face_left_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_place, -rotation_theta, 'left'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_place_left_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_limb, -rotation_theta, 'left'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_limb_left_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_object, -rotation_theta, 'left'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_object_left_sigma6.2.dtseries.nii')

# Right hemisphere
for rotation_theta in np.arange(0,360,10):
    mapping_Right = Make_mapping_vtc2sheet(get_Rvtc_position(plot=False), rotation_theta, hemisphere='right')    
    threshold = 0.5
    warped_face = Mapping_som_area_2_vtc(mapping_Right, face_mask)
    warped_place = Mapping_som_area_2_vtc(mapping_Right, place_mask)
    warped_limb = Mapping_som_area_2_vtc(mapping_Right, limb_mask)
    warped_object = Mapping_som_area_2_vtc(mapping_Right, object_mask)
    plt.figure(figsize=(15,3))
    vtc = make_moving_map(get_Rvtc_position(plot=False), rotation_theta, 'right')
    ax1 = plt.subplot(141)
    ax1.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax1.imshow(warped_face, cmap='Reds', alpha=0.5);ax1.axis('off')
    ax1.set_title(str(rotation_theta))
    ax2 = plt.subplot(142)
    ax2.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax2.imshow(warped_place, cmap='Greens', alpha=0.5);ax2.axis('off')
    ax3 = plt.subplot(143)
    ax3.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax3.imshow(warped_limb, cmap='Oranges', alpha=0.5);ax3.axis('off')
    ax4 = plt.subplot(144)
    ax4.imshow(vtc, cmap='Greys', alpha=1);plt.axis('off')
    ax4.imshow(warped_object, cmap='Blues', alpha=0.5);ax4.axis('off')
    # save
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_face, -rotation_theta, 'right'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_face_right_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_place, -rotation_theta, 'right'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_place_right_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_limb, -rotation_theta, 'right'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_limb_right_sigma6.2.dtseries.nii')
    save_warped_area_as_gii(drive_area_2_vtc_axis(warped_object, -rotation_theta, 'right'), 
                            'D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\warped_object_right_sigma6.2.dtseries.nii')



threshold = 0.5
mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), rotation_theta, hemisphere='left') 
hcp_face,_ = get_L_hcp_space_mask(19, threshold)
warped_face = Mapping_area2sheet(mapping_Left, hcp_face, hemisphere='left')
hcp_place,_ = get_L_hcp_space_mask(20, threshold)
warped_place = Mapping_area2sheet(mapping_Left, hcp_place, hemisphere='left')
hcp_limb,_ = get_L_hcp_space_mask(18, threshold)
warped_limb = Mapping_area2sheet(mapping_Left, hcp_limb, hemisphere='left')
hcp_object,_ = get_L_hcp_space_mask(21, threshold)
warped_object = Mapping_area2sheet(mapping_Left, hcp_object, hemisphere='left')
plt.figure()
plt.imshow(warped_face, 'Reds', alpha=1)
plt.imshow(warped_place, 'Greens', alpha=0.5)
plt.imshow(warped_limb, 'Oranges', alpha=0.3)
plt.imshow(warped_object, 'Blues', alpha=0.4)

mapping_Right = Make_mapping_vtc2sheet(get_Rvtc_position(plot=False), rotation_theta, hemisphere='right') 
hcp_face,_ = get_R_hcp_space_mask(19, threshold)
warped_face = Mapping_area2sheet(mapping_Right, hcp_face, hemisphere='right')
hcp_place,_ = get_R_hcp_space_mask(20, threshold)
warped_place = Mapping_area2sheet(mapping_Right, hcp_place, hemisphere='right')
hcp_limb,_ = get_R_hcp_space_mask(18, threshold)
warped_limb = Mapping_area2sheet(mapping_Right, hcp_limb, hemisphere='right')
hcp_object,_ = get_R_hcp_space_mask(21, threshold)
warped_object = Mapping_area2sheet(mapping_Right, hcp_object, hemisphere='right')
plt.figure()
plt.imshow(warped_face, 'Reds', alpha=1)
plt.imshow(warped_place, 'Greens', alpha=0.5)
plt.imshow(warped_limb, 'Oranges', alpha=0.3)
plt.imshow(warped_object, 'Blues', alpha=0.4)



  

"""Hausdorff Distance"""
###############################################################################
###############################################################################    
def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_maypca(class_name, som, pca_index):  
    '''
    pca_index: a list of features, such as [0,1,2,3]
    '''
    f = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + class_name)
    f.remove('.DS_Store')
    Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
    pca = PCA()
    pca.fit(Data)
    Response_som = []
    for index,pic in enumerate(f):
        img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name+"\\"+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        output = alexnet(picimg).data.numpy()
        #Response_som.append(som.forward_activate(pca.transform(output)[:,pca_index][0]))   
        Response_som.append(1/som.activate(pca.transform(output)[:,pca_index][0]))
    Response_som = np.array(Response_som)
    return Response_som

def Functional_map_pca(som, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    f4.remove('.DS_Store')
    Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
    Data = zscore(Data, axis=0)
    pca = PCA()
    pca.fit(Data)
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
        #Response_som.append(som.forward_activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som[:111,:,:],Response_som[111:172,:,:],Response_som[172:250,:,:],Response_som[250:,:,:]

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask

def get_mask_position(mask):
    return list(zip(np.where(mask!=0)[0], np.where(mask!=0)[1]))
  
def get_som_area(Ws_dir, pca_index):
    for sig in np.round(np.arange(0.1,10.1,0.1),1):
        som = BrainSOM.VTCSOM(200, 200, len(pca_index), sigma=5, learning_rate=1, 
                              neighborhood_function='gaussian', random_seed=0)
        w_dir = Ws_dir + 'som_sigma_' + str(sig) + '.npy'
        som._weights = np.load(w_dir)
            
        Response_face = Functional_map_maypca('face', som, pca_index)
        Response_place = Functional_map_maypca('place', som, pca_index)
        Response_body = Functional_map_maypca('body', som, pca_index)
        Response_object = Functional_map_maypca('object', som, pca_index)
        # Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca_index)
        Response_all = [Response_face,Response_place,Response_body,Response_object]
        save_dir = Ws_dir + 'sigma_' + str(sig) + '_response' + '.npy'
        np.save(save_dir, Response_all)
get_som_area('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\', [0,1,2,3])
    
def hausdorff_Left(w_dir, response_dir, pca_index):
    som = BrainSOM.VTCSOM(200, 200, len(pca_index), sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(w_dir)
    
    Response_all = np.load(response_dir, allow_pickle=True)
    Response_face,Response_place,Response_body,Response_object = Response_all[0],Response_all[1],Response_all[2],Response_all[3]    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
    
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


def hausdorff_Right(w_dir, response_dir, pca_index):
    som = BrainSOM.VTCSOM(200, 200, len(pca_index), sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(w_dir)
    
    Response_all = np.load(response_dir, allow_pickle=True)
    Response_face,Response_place,Response_body,Response_object = Response_all[0],Response_all[1],Response_all[2],Response_all[3]    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold_cohend = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
    place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
    limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
    object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
    
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


def run_hausdorff_in_varied_sigma(Dir, out_file, pca_index):
    with open(out_file, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(['Sigma', 'r_hausdorff_left', 'r_hausdorff_right'])
        for sig in np.round(np.arange(0.1,10.1,0.1),1):
            w_dir = Dir + 'som_sigma_' + str(sig) + '.npy'
            response_dir = Dir + 'sigma_' + str(sig) + '_response' + '.npy'
            r_left = hausdorff_Left(w_dir, response_dir, pca_index)
            r_right = hausdorff_Right(w_dir, response_dir, pca_index)
            csv_write.writerow([str(sig), r_left, r_right])



run_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\', 
                              'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\Sig_hausdorff.csv', 
                              [0,1,2,3])

run_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\\', 
                              'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\Sig_hausdorff.csv', 
                              [0,1,2,3])    
                          






"""Symmetric Diffeomorphic Registration + Hausdorff Distance"""
###############################################################################
###############################################################################
def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_maypca(class_name, som, pca_index):  
    '''
    pca_index: a list of features, such as [0,1,2,3]
    '''
    f = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + class_name)
    f.remove('.DS_Store')
    Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
    pca = PCA()
    pca.fit(Data)
    Response_som = []
    for index,pic in enumerate(f):
        img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name+"\\"+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        output = alexnet(picimg).data.numpy()
        #Response_som.append(som.forward_activate(pca.transform(output)[:,pca_index][0]))   
        Response_som.append(1/som.activate(pca.transform(output)[:,pca_index][0]))
    Response_som = np.array(Response_som)
    return Response_som

def Functional_map_pca(som, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    f4.remove('.DS_Store')
    Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
    Data = zscore(Data, axis=0)
    pca = PCA()
    pca.fit(Data)
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som[:111,:,:],Response_som[111:172,:,:],Response_som[172:250,:,:],Response_som[250:,:,:]

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask

def get_mask_position(mask):
    return list(zip(np.where(mask!=0)[0], np.where(mask!=0)[1]))

def SDR_hausdorff_Left(w_dir, mapping, pca_index, hemisphere):
    som = BrainSOM.VTCSOM(200, 200, 4, sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(w_dir)
        
    Response_face = Functional_map_maypca('face', som, pca_index)
    Response_place = Functional_map_maypca('place', som, pca_index)
    Response_body = Functional_map_maypca('body', som, pca_index)
    Response_object = Functional_map_maypca('object', som, pca_index)
    #Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold)
    place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold)
    limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold)
    object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold)
    
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

def SDR_hausdorff_Right(w_dir, mapping, pca_index, hemisphere):
    som = BrainSOM.VTCSOM(200, 200, 4, sigma=5, learning_rate=1, 
                          neighborhood_function='gaussian', random_seed=0)
    som._weights = np.load(w_dir)
        
    Response_face = Functional_map_maypca('face', som, pca_index)
    Response_place = Functional_map_maypca('place', som, pca_index)
    Response_body = Functional_map_maypca('body', som, pca_index)
    Response_object = Functional_map_maypca('object', som, pca_index)
    #Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca_index)
    
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    threshold = 0.5
    face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold)
    place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold)
    limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold)
    object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold)
    
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


def run_SDR_hausdorff_in_varied_sigma(weights_dir, out_file, mapping_Left, mapping_Right, pca_index):
    f = os.listdir(weights_dir)
    will_be_removed = []
    for i in f:
        if i[:3]!='som':
            will_be_removed.append(i)
    for i in will_be_removed:
        f.remove(i)
    f.sort()
    with open(out_file, 'a+') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(['Sigma', 'r_hausdorff_left', 'r_hausdorff_right'])
        for w_file in tqdm(f):
            w_dir = weights_dir+w_file
            r_left = SDR_hausdorff_Left(w_dir, mapping_Left, pca_index, hemisphere='left')
            r_right = SDR_hausdorff_Right(w_dir, mapping_Right, pca_index, hemisphere='right')
            csv_write.writerow([w_file[10:13], r_left, r_right])



mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), hemisphere='left')
mapping_Right = Make_mapping_vtc2sheet(get_Rvtc_position(plot=False), hemisphere='right')

run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_mypca4_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_mypca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3])

run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_mypca5_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_mypca5_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3,4])

run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca4_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3])     
                       
run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca5_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca5_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3,4])    



run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3])    

run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca5_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca5_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3,4])    

run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3])    
  
run_SDR_hausdorff_in_varied_sigma('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca5_Sigma_200000step\\', 
                                  'D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca5_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                  mapping_Left, mapping_Right, [0,1,2,3,4])    







""""plot"""
###############################################################################
###############################################################################
def plot_r(csv_file, step, medfilt_range):
    x = np.arange(0.1,10.1,0.1)
    x = x[0:100:step]
    file = pd.read_csv(csv_file)
    r_hausdorff_left = file['r_hausdorff_left'].tolist()[0:100:step]
    r_hausdorff_right = file['r_hausdorff_right'].tolist()[0:100:step]
    r_hausdorff_avg = np.array(r_hausdorff_left)/2+np.array(r_hausdorff_right)/2
    r_hausdorff_avg = r_hausdorff_avg
    r_hausdorff_avg = signal.medfilt(r_hausdorff_avg, medfilt_range)
    r_hausdorff_left = signal.medfilt(r_hausdorff_left, medfilt_range)
    r_hausdorff_right = signal.medfilt(r_hausdorff_right, medfilt_range)
    plt.figure(figsize=(10,2))
    plt.subplot(1,3,1)
    plt.plot(x,r_hausdorff_left, color='b')
    plt.title('r_hausdorff_left')
    plt.subplot(1,3,2)
    plt.plot(x,r_hausdorff_right, color='g')
    plt.title('r_hausdorff_right')
    plt.subplot(1,3,3)
    plt.plot(x,r_hausdorff_avg, color='r')
    plt.title('r_hausdorff_avg')
    return r_hausdorff_left, r_hausdorff_right, r_hausdorff_avg


### SDR+Hausdorff
plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_mypca5_Sigma_200000step\Sig_SDR_hausdorff.csv', medfilt_range=5, step=2)   
plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca4_Sigma_200000step\Sig_SDR_hausdorff.csv', medfilt_range=9, step=2) 
plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM(200x200)_pca5_Sigma_200000step\Sig_SDR_hausdorff.csv', medfilt_range=11, step=1)  



r_hausdorff_left, r_hausdorff_right, r_hausdorff_avg = plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\Sig_hausdorff.csv', 
                                                              step=2, medfilt_range=3)    
print(r_hausdorff_left[24:].max(), np.where(r_hausdorff_left==r_hausdorff_left[24:].max())[0]+1, '/',
      r_hausdorff_right[24:].max(), np.where(r_hausdorff_right==r_hausdorff_right[24:].max())[0]+1, '/',
      r_hausdorff_avg[24:].max(), np.where(r_hausdorff_avg==r_hausdorff_avg[24:].max())[0]+1)

r_hausdorff_left, r_hausdorff_right, r_hausdorff_avg = plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\\Sig_hausdorff.csv', 
                                                              step=1, medfilt_range=3)    
print(r_hausdorff_left[24:].max(), np.where(r_hausdorff_left==r_hausdorff_left[24:].max())[0]+1, '/',
      r_hausdorff_right[24:].max(), np.where(r_hausdorff_right==r_hausdorff_right[24:].max())[0]+1, '/',
      r_hausdorff_avg[24:].max(), np.where(r_hausdorff_avg==r_hausdorff_avg[24:].max())[0]+1)



r_hausdorff_left, r_hausdorff_right, r_hausdorff_avg = plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_mypca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                                              step=1, medfilt_range=7)   
number = 25  
print(r_hausdorff_left[number:].max(), np.where(r_hausdorff_left==r_hausdorff_left[number:].max())[0]+1, '/',
      r_hausdorff_right[number:].max(), np.where(r_hausdorff_right==r_hausdorff_right[number:].max())[0]+1, '/',
      r_hausdorff_avg[number:].max(), np.where(r_hausdorff_avg==r_hausdorff_avg[number:].max())[0]+1)

r_hausdorff_left, r_hausdorff_right, r_hausdorff_avg = plot_r('D:\\TDCNN\Results\Alexnet_fc8_SOM\SOM_norm(200x200)_pca4_Sigma_200000step\\New_Sig_SDR_hausdorff.csv', 
                                                              step=1, medfilt_range=3)  
number = 50  
print(r_hausdorff_left[number:].max(), np.where(r_hausdorff_left==r_hausdorff_left[number:].max())[0]+1, '/',
      r_hausdorff_right[number:].max(), np.where(r_hausdorff_right==r_hausdorff_right[number:].max())[0]+1, '/',
      r_hausdorff_avg[number:].max(), np.where(r_hausdorff_avg==r_hausdorff_avg[number:].max())[0]+1)


