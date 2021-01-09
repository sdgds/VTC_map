#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import sys
sys.path.append('/home/dell/Desktop/git_item/dnnbrain/')
from dnnbrain.dnn import core


'1. How to convolution the HRF function for DNN'
def get_cnn_hrf_signal(actdata, tr=2, fps=1.0/30):
    timept = actdata.shape[0]
    actlength = (timept)*(fps)
    onset = np.linspace(0, (timept)*fps, timept)
    duration = np.array([1.0/30]*timept)    
    cnn_hrf_signal = core.convolve_hrf(actdata, onset, duration, int(actlength/tr), tr)[1:]
    return cnn_hrf_signal

actdata = np.load('/home/dell/Desktop/DNN2Brain/Conv5_data/Alexnet_seg10.pkl', allow_pickle=True)
actdata = np.vstack((actdata[184], actdata[124]))
actdata[actdata<0] = 0
actdata = np.log(actdata+0.01)  
hrf_signal = get_cnn_hrf_signal(actdata.T)
hrf_signal = stats.zscore(hrf_signal)
    

'2. How to get index of vertex data(from wen to 32K space)'
# This function can convert mask in 32K space to Wen's data(32K has 32491 vertex, but Wen has 29695 vertex)
def Mask_mapping_32k2wen(mask_32k, Dict_32k_to_wen):
    mask_wen = np.zeros(len(Dict_32k_to_wen))
    for i in list(Dict_32k_to_wen.keys()):
        mask_wen[Dict_32k_to_wen[i]] = mask_32k[i]
    mask_wen = np.where(mask_wen!=0, 1, 0)
    return mask_wen

# Get left and right cortex
seg1_1_Atlas = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
list_of_block = list(seg1_1_Atlas.header.get_index_map(1).brain_models)
for i in range(len(list_of_block)):
    print(list_of_block[i].brain_structure)
    print('index_offset:', list_of_block[i].index_offset)
    print('index_count:', list_of_block[i].index_count)
    print('-----------------------------------')
CORTEX_Left = list_of_block[0]
CORTEX_Right = list_of_block[1]

# Make the mapping from 32K to Wen for left and right cortex
Dict_wen_to_32kL = dict()
for vertex in range(CORTEX_Left.index_count):
    Dict_wen_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
Dict_32kL_to_wen = {v:k for k,v in Dict_wen_to_32kL.items()}
Dict_wen_to_32kR = dict()
for vertex in range(CORTEX_Right.index_count):
    Dict_wen_to_32kR[vertex] = CORTEX_Right.vertex_indices[vertex]
Dict_32kR_to_wen = {v:k for k,v in Dict_wen_to_32kR.items()}

# Use Zhen's functinoal mask, 200 vertex are gained in every ROI
ther = -200
mask_lofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-lOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_lofa[np.argsort(mask_lofa)[ther:]] = 1
mask_lofa = np.where(mask_lofa==1, 1, 0)                  # OFA mask in 32K space
mask_lofa_wen = Mask_mapping_32k2wen(mask_lofa, Dict_32kL_to_wen)              # Mapping 32K mask to Wen's data
mask_rofa = nib.load('/home/dell/Desktop/DNN2Brain/Mask/Face/32K/BAA-OR-FvO-rOFA-PRM-fsaverage.func.gii').darrays[0].data
mask_rofa[np.argsort(mask_rofa)[ther:]] = 1
mask_rofa = np.where(mask_rofa==1, 1, 0)
mask_rofa_wen = Mask_mapping_32k2wen(mask_rofa, Dict_32kR_to_wen)


data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
R2map_alexnet[:29696] = np.multiply(R2map_alexnet[:29696], mask_lofa_wen)
img = np.array([])
for i in range(245):
    img = np.append(img, z)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/test.dtseries.nii')


## many useful mask in MMP
mask = nib.load('/home/dell/Desktop/DNN2Brain/Mask/MMP/surface/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
mask_ffa = np.where((mask==18)|(mask==198))[0]
mask_ofa = np.where((mask==22)|(mask==202))[0]
mask_sts = np.where((mask==128)|(mask==129)|(mask==130)|(mask==308)|(mask==309)|(mask==310))[0]
mask_loc = np.where((mask==20)|(mask==21)|(mask==200)|(mask==201))[0]
mask_tpoj = np.where((mask==140)|(mask==141)|(mask==320)|(mask==321))[0]

Zero = np.zeros(91282)
Zero[mask_ffa] = 1
Zero[mask_ofa] = 2
Zero[mask_sts] = 3
Zero[mask_loc] = 4
Zero[mask_tpoj] = 5
Mask_Face = Zero

data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
img = np.array([])
for i in range(245):
    img = np.append(img, Mask_Face)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/Mask_Face.dtseries.nii')


## make mask in Face and Object
mask = nib.load('/home/dell/Desktop/DNN2Brain/MMP_mask/surface/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
HCP_data = nib.load('/home/dell/Desktop/DNN2Brain/HCP/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask_face = np.where((mask==150)|(mask==151)|(mask==139)|(mask==129)|(mask==18)|(mask==22)|(mask==150+180)|(mask==140+180)|(mask==141+180)|(mask==22+180)|(mask==139+180)|(mask==2+180)|(mask==128+180)|(mask==129+180)|(mask==132+180)|(mask==18+180))[0]
hcp_face = HCP_data.dataobj[19,:]
Zero = np.zeros(91282)
Zero[mask_face] = 1
hcp_face = np.where(Zero==1, hcp_face, 0)
hcp_face = np.where(hcp_face>=0.3, hcp_face, 0)
data_nii = nib.load('/home/dell/Desktop/DNN2Brain/video_fmri_dataset/subject1/fmri/seg1/cifti/seg1_1_Atlas.dtseries.nii')
img = np.array([])
for i in range(245):
    img = np.append(img, hcp_face)
img = img.reshape(245, -1)
IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
nib.save(IMG, '/home/dell/Desktop/DNN2Brain/Results_map/HCP_Face.dtseries.nii')


