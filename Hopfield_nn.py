# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import csv
import copy
import scipy.io as scio
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
import nibabel as nib
import scipy.stats as stats
import networkx as nx
from scipy.integrate import odeint
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import itertools
import sys
sys.path.append('D:\\TDCNN')
import BrainSOM
import Hopfield_VTCSOM
import Generative_adv_picture


### Data
def bao_preprocess_pic(img):
    img = img.resize((224,224))
    img = np.array(img)-237.169
    picimg = torch.Tensor(img).permute(2,0,1)
    return picimg

data_transforms = {
    'see': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'see_flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1)]),
    'flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
    }
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()


def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_pca(som, pca, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    mean_features = np.mean(Response, axis=0)
    std_features = np.std(Response, axis=0)
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som, (mean_features, std_features)

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1])) - 1
    Cohend = []
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            Cohend.append(cohend)
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                mask[i,j] = 1
    return mask  
    
def Picture_activation(pic_dir, som, pca, pca_index, mean_features, std_features, mask=None):
    """"mask is like (3,224,224)"""
    img = Image.open(pic_dir).convert('RGB')
    if mask!=None:
        picimg = data_transforms['val'](img) * mask
    else:
        picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    if mask!=None:
        img_mask_see = np.multiply(img_see, mask.permute(1,2,0).data.numpy())
    else:
        img_mask_see = img_see
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, img_mask_see, response_som

def Picture_activation(pic_dir, som, pca, pca_index, mean_features, std_features, mask_method, severity):
    """"mask_method is like Generative_adv_picture.gaussian_noise"""
    img = Image.open(pic_dir).convert('RGB')#.resize((224,224))
    img_adv = mask_method(img, severity=severity)
    img_adv = Image.fromarray(np.uint8(img_adv))
    picimg = data_transforms['val'](img_adv)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    img_mask_see = np.array(data_transforms['see'](img_adv))
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, img_mask_see, output, response_som

def Pure_picture_activation(pic_dir, prepro_method, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB')
    if prepro_method=='val':
        picimg = data_transforms['val'](img)
        img_see = np.array(data_transforms['see'](img))
    if prepro_method=='flip':
        picimg = data_transforms['flip'](img)
        img_see = np.array(data_transforms['see_flip'](img))
    if prepro_method=='bao':
        picimg = bao_preprocess_pic(img)
        img_see = np.array(data_transforms['val'](img))        
    picimg = picimg.unsqueeze(0) 
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som

def Upset_picture_activation(pic_dir, block_num_row, som, pca, pca_index, mean_features, std_features):
    img = Image.open(pic_dir).convert('RGB').resize((224,224))
    img = np.array(img)
    block_num_row = np.uint8(block_num_row)
    t = np.uint8(224/block_num_row)
    for time in range(1000):
        left_up_row_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_1 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_1 = left_up_row_1+t
        right_down_col_1 = left_up_col_1+t
        temp = copy.deepcopy(img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1, :])
        left_up_row_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        left_up_col_2 = np.random.choice(np.arange(0,block_num_row,1),1).item()*t
        right_down_row_2 = left_up_row_2+t
        right_down_col_2 = left_up_col_2+t
        img[left_up_row_1:right_down_row_1, left_up_col_1:right_down_col_1,:] = img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:]
        img[left_up_row_2:right_down_row_2, left_up_col_2:right_down_col_2,:] = temp
    img = Image.fromarray(img)
    picimg = data_transforms['val'](img)
    picimg = picimg.unsqueeze(0) 
    img_see = np.array(data_transforms['see'](img))
    output = alexnet(picimg).data.numpy()
    response = (output-mean_features)/std_features
    response_som = 1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index])
    return img_see, output, response_som
    
def plot_memory(img, initial_state, state, memory_pattern):
    plt.figure(figsize=(23,4))
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Original picture');plt.axis('off')
    plt.subplot(142)
    plt.imshow(initial_state)
    plt.title('Initial state');plt.axis('off')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(state)
    plt.title('Stable state');plt.axis('off')
    plt.colorbar() 
    plt.subplot(144)
    plt.imshow(memory_pattern)
    plt.title('right state');plt.axis('off')
    plt.colorbar() 
    
                
                
                
### sigma=6.2
som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian', random_seed=0)
som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_6.2.npy')
#som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_5.0.npy')

Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
Data = zscore(Data)
pca = PCA()
pca.fit(Data)
Response_som, (mean_features,std_features) = Functional_map_pca(som, pca, [0,1,2,3])
Response_face = Response_som[:111,:,:]
Response_place = Response_som[111:172,:,:]
Response_body = Response_som[172:250,:,:]
Response_object = Response_som[250:,:,:]
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
threshold_cohend = 0.5
face_mask = som_mask(som, Response_face, Contrast_respense, 0, threshold_cohend)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, threshold_cohend)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, threshold_cohend)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, threshold_cohend)
training_pattern = np.array([face_mask.reshape(-1),
                             place_mask.reshape(-1),
                             limb_mask.reshape(-1),
                             object_mask.reshape(-1)])


model = Hopfield_VTCSOM.Stochastic_Hopfield_nn(x=200, y=200, pflag=1, nflag=-1,
                                               patterns=[face_mask,place_mask,limb_mask,object_mask])
# model.reconstruct_w([training_pattern])
model.reconstruct_w_with_structure_constrain([training_pattern], 'exponential', 0.023) # Human(0.0238)
# model.reconstruct_w_with_structure_constrain([training_pattern], 'exponential', 0.042) # Macaque(0.0416)




"1. associative inference"
###############################################################################
External_field_prior = np.zeros((200,200))

# How many epoches?
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.ones((3,224,224))
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                    mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
Delta = []
for epochs in [100000,150000,200000,250000,300000]:
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=0.1,
                                    H_prior=External_field_prior, H_lik=False, epochs=epochs)
    Delta.append(model.order_parameter(stable_state, training_pattern[0]))
plt.plot(np.abs(Delta))


# Face
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.zeros((3,224,224))
mask[:,:135,:] = 1
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                    mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                H_prior=External_field_prior, H_bottom_up=initial_state, epochs=200000, save_inter_step=1000)
plot_memory(img_mask_see, initial_state, stable_state[0].reshape(200,200), 
            training_pattern[0].reshape(200,200))
## visulization
model.dynamics_pattern('Half_face.gif', model.dynamics_state)
Dynamic_states = model.dynamics_state
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()



pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
img_see, img_mask_see, alexnet_output, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                                          mean_features, std_features, Generative_adv_picture.pixelate, severity=4)
Imgnet_label = np.load('D:\\TDCNN\\Imgnet_label.npy', allow_pickle=True).tolist()
plt.figure()
plt.subplot(121)
plt.imshow(img_mask_see);plt.axis('off')
plt.title('Alexnet prediction is: %s' %Imgnet_label[alexnet_output[0].argmax()])
plt.subplot(122)
plt.imshow(initial_state);plt.axis('off')

initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                H_prior=External_field_prior, H_lik=False, epochs=250000, save_inter_step=1000)
plot_memory(img_mask_see, initial_state, stable_state.reshape(200,200), 
            training_pattern[0].reshape(200,200))


# Object
pic_dir = 'D://TDCNN//HCP//HCP_WM//object/TO_054_TOOL54_BA.png'
mask = torch.ones((3,224,224)).int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                    mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                         H_prior=External_field_prior, H_lik=False, epochs=200000)
plot_memory(img_mask_see, initial_state, stable_state[0].reshape(200,200), 
            training_pattern[3].reshape(200,200))

initial_state = np.zeros((200,200))
initial_state[0:50,:] = 1
initial_state = np.where(initial_state==0, -1, 1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                         H_prior=External_field_prior, H_lik=False, epochs=200000)
plot_memory(initial_state, initial_state, stable_state[0].reshape(200,200), 
            training_pattern[3].reshape(200,200))




"2. Beta: phase transition"
###############################################################################
External_field_prior = np.zeros((200,200))
#External_field_prior[50:75,50:75] = 10

pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.zeros((3,224,224))
mask[:,:150,:] = 1
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                          mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)

Beta_state_dict = dict()
for beta in np.round(np.arange(0,51,1),1):
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], 
                                          beta=beta, H_prior=External_field_prior, 
                                          H_lik=False, epochs=250000)
    plot_memory(img_mask_see, initial_state, stable_state.reshape(200,200), 
                training_pattern[0].reshape(200,200))
    Beta_state_dict[beta] = stable_state[0].reshape(200,200)
    
np.save('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\\Hopfield_nn\\Beta_state_dict_6.2.npy',
        Beta_state_dict)
    

Beta_state_dict = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\\Hopfield_nn\\Beta_state_dict_9.3.npy',
                          allow_pickle=True).item()
Delta_face = []
Delta_place = []
Delta_body = []
Delta_object = []
for beta in Beta_state_dict.keys():
    stable_state = Beta_state_dict[beta].reshape(40000)
    Delta_face.append(model.order_parameter(stable_state, training_pattern[0]))
    Delta_place.append(model.order_parameter(stable_state, training_pattern[1]))
    Delta_body.append(model.order_parameter(stable_state, training_pattern[2]))
    Delta_object.append(model.order_parameter(stable_state, training_pattern[3]))

plt.figure()
plt.plot(Delta_face, label='face', color='red')
plt.plot(Delta_place, label='place', color='green')
plt.plot(Delta_body, label='body', color='yellow')
plt.plot(Delta_object, label='object', color='blue')
plt.legend()
plt.xlabel('Beta')
plt.ylabel('Accuracy')




"3. 92 images"
###############################################################################
def Get_mean_std(): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'face')
    f1.remove('.DS_Store')
    f2 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'place')
    f2.remove('.DS_Store')
    f3 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'body')
    f3.remove('.DS_Store')
    f4 = os.listdir("D:\\TDCNN\HCP\HCP_WM\\" + 'object')
    f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("D:\\TDCNN\HCP\HCP_WM\\"+class_name[index]+"\\"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    return Response.mean(axis=0), Response.std(axis=0)
mean, std = Get_mean_std()

External_field_prior = np.zeros((200,200))
files = os.listdir(r'D:\TDCNN\Data\algonautsChallenge2019\Training_Data\92_Image_Set\92images\\')

Image_state_dict = dict()
for index,f in enumerate(files):
    pic_dir = r'D:\TDCNN\Data\algonautsChallenge2019\Training_Data\92_Image_Set\92images\\' + f
    img_see, img_mask_see, initial_state = Pure_picture_activation(pic_dir, som, pca, [0,1,2,3], mean, std)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100, 
                                             H_prior=External_field_prior, H_bottom_up=True, 
                                             epochs=250000, save_inter_step=1000)
    plot_memory(img_see, initial_state, stable_state.reshape(200,200), training_pattern[3].reshape(200,200))
    Image_state_dict[index] = stable_state[0].reshape(200,200)
    
images_92_response = []
for k,v in Image_state_dict.items():
    images_92_response.append(v)
images_92_response = np.array(images_92_response)

plt.figure()
plt.imshow(1-np.corrcoef(images_92_response.reshape(92,-1)), cmap='jet')
plt.colorbar()
    




"4. Dynamics + Prior"
###############################################################################
External_field_prior = np.zeros((200,200))
External_field_prior[0:40,:] = 10
External_field_prior[180:,150:] = 10
External_field_prior = np.where(External_field_prior==0, -10, 10)
# External_field_prior = copy.deepcopy(face_mask)*10
# External_field_prior = copy.deepcopy(limb_mask)*10

pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
mask = torch.zeros((3,224,224))
mask[:,:135,:] = 1
mask = mask.int()
img_see, img_mask_see, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                    mean_features, std_features, mask)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=50,
                                H_prior=External_field_prior, H_lik=False, epochs=200000)
plot_memory(img_mask_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))


pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
img_see, img_mask_see, alexnet_output, initial_state = Picture_activation(pic_dir, som, pca, [0,1,2,3], 
                                                                          mean_features, std_features, Generative_adv_picture.pixelate, severity=1)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                H_prior=External_field_prior, H_lik=False, epochs=200000)
plot_memory(img_mask_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))





"5. Dynamics + Prior (face vase illusion)"
###############################################################################
### Face Vase illusion
## top down (initial steps)
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_vase_illusion_1.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((object_mean_act-face_mean_act)/(face_mean_act+object_mean_act), label='(obj-face)/(face+obj)')
plt.plot((object_mean_act-face_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(obj-face-body-place)/(face+obj+body+place)')
plt.legend()



## top down (initial steps)
External_field_prior = copy.deepcopy(face_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_vase_illusion_2.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()




### Face stimuli + Object top down
## top down (initial steps)
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/FC_009_M5.png'
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Face_stim_obj_topdown.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((face_mean_act-object_mean_act)/(face_mean_act+object_mean_act), label='(face-obj)/(face+obj)')
plt.plot((face_mean_act-object_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(face-obj-body-place)/(face+obj+body+place)')
plt.legend()


### Object stimuli + Face top down
## top down (initial steps)
External_field_prior = copy.deepcopy(face_mask) * 2
pic_dir = 'D://TDCNN//HCP//HCP_WM//object/TO_076_TOOL76_BA.png'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                          H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
## visulization
model.dynamics_pattern('Object_stim_face_topdown.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot((object_mean_act-face_mean_act)/(face_mean_act+object_mean_act), label='(obj-face)/(face+obj)')
plt.plot((object_mean_act-face_mean_act-body_mean_act-place_mean_act)/(face_mean_act+object_mean_act+body_mean_act+place_mean_act), label='(obj-face-body-place)/(face+obj+body+place)')
plt.legend()



### Random stimuli + bottom up
# bottom up (initial steps)
mean_stable_state = np.zeros((200,200))
pic_list = os.listdir('D://TDCNN//HCP//HCP_WM//face/')
pic_list.remove('.DS_Store')
for i in range(30):
    External_field_prior = np.zeros((200,200))
    pic_dir = 'D://TDCNN//HCP//HCP_WM//face/' + np.random.choice(pic_list,1)[0]
    img_see, alexnet_output, initial_state = Upset_picture_activation(pic_dir, 4, som, pca, [0,1,2,3], mean_features, std_features)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                              epochs=160000, save_inter_step=1000)
    plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
    mean_stable_state += stable_state[0].reshape(200,200)
mean_stable_state = mean_stable_state/30


### Random stimuli + bottom up + top down
External_field_prior = copy.deepcopy(face_mask) * 2
pic_list = os.listdir('D://TDCNN//HCP//HCP_WM//face/')
pic_list.remove('.DS_Store')
mean_stable_state = np.zeros((200,200))
for i in range(30):
    pic_dir = 'D://TDCNN//HCP//HCP_WM//face/' + np.random.choice(pic_list,1)[0]
    img_see, alexnet_output, initial_state = Upset_picture_activation(pic_dir, 4, som, pca, [0,1,2,3], mean_features, std_features)
    initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
    stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
    ## dynamics only (posterior steps)
    External_field_prior = np.zeros((200,200))
    stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                              H_prior=External_field_prior, H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
    plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
    mean_stable_state += stable_state[0].reshape(200,200)
mean_stable_state = mean_stable_state/30






"6. Dynamics with changed beta"
###############################################################################
def sigmoid(data):
    return 1/(1+np.exp(-10*(data-0.4)))

def column_segmentation(threshold):
    U = som.U_avg_matrix()
    U_copy = copy.deepcopy(U)
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



## 只通过bottom up的切换和临界态的过渡，能完成状态切换
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = face_mask*2
initial_state = np.where(face_mask+object_mask==0, 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta,
                                         H_prior=External_field_prior, H_bottom_up=initial_state, 
                                         epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def changed_beta_func(beta, t, max_iter):
    return (70 / (1+t/(max_iter/20)))-50
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, mask=face_mask,
                                                              changed_beta_func=changed_beta_func,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, mask=object_mask,
                                                              changed_beta_func=changed_beta_func,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## 无top down,临界态过渡，能完成状态切换
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = np.zeros((200,200))
initial_state = np.where(face_mask+object_mask==0, 1, -1)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([initial_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def face_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def object_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## 状态切换可能需要bottom up不能太差
beta = 15
top_down_time = 80000
no_top_down_time = 80000
save_inter_step = 10000

## top down (face)
model.rebuild_up_param()
External_field_prior = face_mask*2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
def face_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def object_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')
## dynamics only (posterior steps)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([stable_state.reshape(-1)], beta=beta, 
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=no_top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state));plt.colorbar();plt.axis('off')

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()



## beta(perception)
def changed_beta_func(region_avg_dynamics_state, beta, tao, adaptation_lower_bound=2):
    """
    beta: a number
    tao: a number
    adaptation_lower_bound: a number
    """
    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0]>=2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp
    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp
    def beta_first_order_difference(r_1):
        if r_1>0.03:
            return 15
        else:
            return adaptation_lower_bound
    def beta_r(r):
        return (15-adaptation_lower_bound)/(1+np.exp(100*(r-0.5))) + adaptation_lower_bound
    r = []
    r_1 = []
    for i in range(1,tao+1):
        r.append(get_firing_rate(region_avg_dynamics_state, -i))
        r_1.append(first_order_difference(region_avg_dynamics_state, -i, -(i+1)))
    r_1 = np.mean(r_1)
    r = np.mean(r)
    if np.abs(r_1) < 0.03:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
    
beta = 15
top_down_time = 80000
no_top_down_time = 600000
save_inter_step = 40000

## top down
model.rebuild_up_param()
External_field_prior = np.zeros((200,200))
initial_state = np.where(face_mask+object_mask==0, 1, -1)
def object_changed_beta_func(beta, t, max_iter):
    return (15 / (1+t/(max_iter/20)))
def face_changed_beta_func(beta, t, max_iter):
    return (1/(1+np.exp(-0.001*t))) * (np.exp(-0.0001*t)+28) - 13
def other_changed_beta_func(beta, t, max_iter):
    return 15
mask_region_beta = dict()
mask_region_beta['face_mask'] = face_changed_beta_func
mask_region_beta['place_mask'] = other_changed_beta_func
mask_region_beta['limb_mask'] = other_changed_beta_func
mask_region_beta['object_mask'] = object_changed_beta_func
stable_state = model.stochastic_dynamics_changed_beta_in_mask([initial_state.reshape(-1)], beta=beta,
                                                              mask_region_beta=mask_region_beta,
                                                              H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                              epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], 
                                                      beta=beta, changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()

plt.figure(dpi=300)
face_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, face_mask)
object_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, object_mask)
plt.plot(face_mean_act, label='Avg betas in face mask')
plt.plot(object_mean_act, label='Avg betas in object mask')
plt.legend()



## face vase double transition
def changed_beta_func(region_avg_dynamics_state, beta, tao):
    """
    beta: a number
    tao: a number
    """
    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0]>=2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp
    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp
    def beta_first_order_difference(r_1):
        if r_1>0.03:
            return 15
        else:
            return 2   
    def beta_r(r):
        return 13/(1+np.exp(100*(r-0.5)))+2
    r = []
    r_1 = []
    for i in range(1,tao+1):
        r.append(get_firing_rate(region_avg_dynamics_state, -i))
        r_1.append(first_order_difference(region_avg_dynamics_state, -i, -(i+1)))
    r_1 = np.mean(r_1)
    r = np.mean(r)
    if np.abs(r_1) < 0.03:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
    
beta = 15
top_down_time = 80000
no_top_down_time = 300000
save_inter_step = 40000
diff_map = column_segmentation(0.00115)
columns_pos_dict = Column_position_set(diff_map)

## top down (face)
model.rebuild_up_param()
External_field_prior = copy.deepcopy(object_mask) * 2
pic_dir = 'C:\\Users\\12499\\Desktop\\Hopfiled_SOM\\face_vase_2.jpg'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics_changed_beta([initial_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
#_,random_bottom_up_1 = search_column_from_seed(columns_pos_dict, (9,64))
#_,random_bottom_up_2 = search_column_from_seed(columns_pos_dict, (57,2))
#_,random_bottom_up_3 = search_column_from_seed(columns_pos_dict, (75,48))
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()

plt.figure(dpi=300)
face_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, face_mask)
object_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, object_mask)
body_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, limb_mask)
place_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, place_mask)
plt.plot(face_mean_act, label='Avg betas in face mask')
plt.plot(object_mean_act, label='Avg betas in object mask')
plt.plot(place_mean_act, label='Avg betas in place mask')
plt.plot(body_mean_act, label='Avg betas in body mask')
plt.legend()



## Multistable state
beta = 15
no_top_down_time = 200000
save_inter_step = 40000

model.rebuild_up_param()
External_field_prior = np.zeros((200,200))
pic_dir = 'D://TDCNN//HCP//HCP_WM//face/f100.bmp'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics_changed_beta([initial_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[0].reshape(200,200))
Dynamic_states = model.dynamics_state

pic_dir = 'D://TDCNN//HCP//HCP_WM//object/TO_076_TOOL76_BA.png'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[3].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

pic_dir = 'D://TDCNN//HCP//HCP_WM//body/BP_034_FLPUI_BA.png'
img_see, alexnet_output, initial_state = Pure_picture_activation(pic_dir, 'val', som, pca, [0,1,2,3], mean_features, std_features)
initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], beta=beta, 
                                                      changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
plot_memory(img_see, initial_state, stable_state[0].reshape(200,200), training_pattern[2].reshape(200,200))
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

model.dynamics_pattern('Multistable_state.gif', Dynamic_states)
plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()






"5. Similar face pictures"
###############################################################################
from PIL import ImageGrab
 
for i in tqdm(range(10)):
    img = ImageGrab.grab(bbox=(150, 250, 1750, 1850)).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    output = alexnet(picimg).cpu().data.numpy().reshape(-1)
    output = (output-mean_features)/std_features
    output_pca = pca.transform(output.reshape(1,-1))[:,range(4)]
    response = 1/som.activate(output_pca)
    if i == 9:
        img = ImageGrab.grab(bbox=(150, 250, 1750, 1850)).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0)
        output = alexnet(picimg).cpu().data.numpy().reshape(-1)
        output = (output-mean_features)/std_features
        output_pca = pca.transform(output.reshape(1,-1))[:,range(4)]
        response = 1/som.activate(output_pca)
        plt.figure(dpi=300)
        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(response)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(np.where(response>np.percentile(response,50),1,0))
        plt.axis('off')
img.save('D:/DNN_Stimulus/Face_similar_pictures/head_0/head_1.png')
img.save('D:/DNN_Stimulus/Face_similar_pictures/head_90/head_1.png')
img.save('D:/DNN_Stimulus/Face_similar_pictures/head_135/head_1.png')
img.save('D:/DNN_Stimulus/Face_similar_pictures/head_180/head_1.png')
            
def normalize(x):
    x = np.array(x)
    x_normalized = (x-x.min())/(x.max()-x.min())
    return x_normalized

def pattern_stabel_state(piclist_dir, prepro_method, EFP, out_type):
    def avg_activation_in_mask(pattern, mask):
        return pattern[np.where(mask==1)].mean()
    piclist = os.listdir(piclist_dir)
    # piclist = np.random.choice(piclist,1).tolist()
    Out_put = []
    Out_put_ = []
    for picdir in piclist:
        picdir = piclist_dir + picdir
        img_see, alexnet_output, initial_state = Pure_picture_activation(picdir, prepro_method, som, pca, [0,1,2,3], mean_features, std_features)
        model.rebuild_up_param()
        if out_type=='som_response':
            som_act = copy.deepcopy(initial_state)
            out = avg_activation_in_mask(som_act, face_mask)
            out_ = avg_activation_in_mask(som_act, object_mask) 
            plt.figure()
            plt.imshow(som_act)
        if out_type=='initial_response':
            initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
            out = avg_activation_in_mask(initial_state, face_mask)
            out_ = avg_activation_in_mask(initial_state, object_mask)
        if out_type=='hopfield_response':
            initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
            stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                                     H_prior=np.zeros((200,200)), H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
            Dynamic_states = model.dynamics_state
            out = (avg_activation_in_mask(Dynamic_states[-1], face_mask)+1)/2
            out_ = (avg_activation_in_mask(Dynamic_states[-1], object_mask)+1)/2
        if out_type=='top_down_hopfield_response':
            initial_state = np.where(initial_state>np.percentile(initial_state,50), 1, -1)
            stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=100,
                                                     H_prior=EFP, H_bottom_up=initial_state, epochs=80000, save_inter_step=1000)
            Dynamic_states = model.dynamics_state
            stable_state = model.stochastic_dynamics([stable_state.reshape(-1)], beta=100,
                                                     H_prior=np.zeros((200,200)), H_bottom_up=initial_state, epochs=160000, save_inter_step=1000)
            Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))
            out = (avg_activation_in_mask(Dynamic_states[-1], face_mask)+1)/2
            out_ = (avg_activation_in_mask(Dynamic_states[-1], object_mask)+1)/2
        Out_put.append(out)
        Out_put_.append(out_)
    Out_put = np.array(Out_put)
    Out_put_ = np.array(Out_put_)
    return (Out_put-Out_put_)/(Out_put+Out_put_), (Out_put,Out_put_)

EFP = np.zeros((200,200))
a,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/face/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
b,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/animal_face/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
c,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/schematic_face/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
d,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/object/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
A,B,C,D = normalize([a.mean(),b.mean(),c.mean(),d.mean()])
sd = [a.std()/np.sqrt(20),b.std()/np.sqrt(20),c.std()/np.sqrt(20),d.std()/np.sqrt(20)]
plt.figure(dpi=300)
plt.bar([0,0.5,1,1.5], height=[A,B,C,D], yerr=sd, width=0.4, label='in face region');plt.legend()
plt.xticks([0,0.5,1,1.5], ['face', 'animal face', 'schematic', 'object'], rotation=20)
plt.text(0, A+0.001, '%.4f' % A, ha='center', va='bottom')
plt.text(0.5, B+0.001, '%.4f' % B, ha='center', va='bottom')
plt.text(1, C+0.001, '%.4f' % C, ha='center', va='bottom')
plt.text(1.5, D+0.001, '%.4f' % D, ha='center', va='bottom')


EFP = np.zeros((200,200))
a,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/cartoon/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
b,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/cartoon/', prepro_method='flip', EFP=EFP, out_type='hopfield_response')
c,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/no_eyes/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
d,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/eyes/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
A,B,C,D = normalize([a.mean(),b.mean(),c.mean(),d.mean()])
sd = [a.std()/np.sqrt(10),b.std()/np.sqrt(10),c.std()/np.sqrt(20),d.std()/np.sqrt(20)]
plt.figure(dpi=300)
plt.bar([0,0.5,1,1.5], height=[A,B,C,D], yerr=sd, width=0.4, label='in face region');plt.legend()
plt.xticks([0,0.5,1,1.5], ['cartoon', 'inverted cartoon', 'no_eyes', 'eyes'], rotation=20)
plt.text(0, A+0.001, '%.4f' % A, ha='center', va='bottom')
plt.text(0.5, B+0.001, '%.4f' % B, ha='center', va='bottom')
plt.text(1, C+0.001, '%.4f' % C, ha='center', va='bottom')
plt.text(1.5, D+0.001, '%.4f' % D, ha='center', va='bottom')


EFP = np.zeros((200,200))
a,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/head_0/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
b,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/head_90/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
c,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/head_135/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
d,_ = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/head_180/', prepro_method='val', EFP=EFP, out_type='hopfield_response')
A,B,C,D = normalize([a.mean(),b.mean(),c.mean(),d.mean()])
sd = [a.std()/np.sqrt(20),b.std()/np.sqrt(20),c.std()/np.sqrt(20),d.std()/np.sqrt(20)]
plt.figure(dpi=300)
plt.bar([0,0.5,1,1.5], height=[A,B,C,D], yerr=sd, width=0.4, label='in face region');plt.legend()
plt.xticks([0,0.5,1,1.5], ['head_0', 'head_90', 'head_135', 'head_180'], rotation=20)
plt.text(0, A+0.001, '%.4f' % A, ha='center', va='bottom')
plt.text(0.5, B+0.001, '%.4f' % B, ha='center', va='bottom')
plt.text(1, C+0.001, '%.4f' % C, ha='center', va='bottom')
plt.text(1.5, D+0.001, '%.4f' % D, ha='center', va='bottom')


# top down is the cause of decay in EEG signal
EFP = copy.deepcopy(face_mask) * 2
Out_schematic,(a_schematic,b_schematic) = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/schematic_face/', prepro_method='val', EFP=EFP, out_type='top_down_hopfield_response')
Out_noneyes,(a_noneyes,b_noneyes) = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/no_eyes/', prepro_method='val', EFP=EFP, out_type='top_down_hopfield_response')
Out_eyes,(a_eyes,b_eyes) = pattern_stabel_state(piclist_dir=r'D:/DNN_Stimulus/Face_similar_pictures/eyes/', prepro_method='val', EFP=EFP, out_type='top_down_hopfield_response')






"6. Structural constraints modulate the equilibrium between steady state and metastable states"
###############################################################################
def changed_beta_func(region_avg_dynamics_state, beta, tao, adaptation_lower_bound=2):
    def get_firing_rate(dynamics_state, t):
        if dynamics_state.shape[0]>=2:
            temp = dynamics_state[t]
        else:
            temp = dynamics_state.mean(axis=0)
        return temp
    def first_order_difference(dynamics_state, t2, t1):
        temp = get_firing_rate(dynamics_state, t2) - get_firing_rate(dynamics_state, t1)
        return temp
    def beta_first_order_difference(r_1):
        if r_1>0.03:
            return 15
        else:
            return adaptation_lower_bound 
    def beta_r(r):
        return (15-adaptation_lower_bound)/(1+np.exp(100*(r-0.5)))+adaptation_lower_bound
    r = []
    r_1 = []
    for i in range(1,tao+1):
        r.append(get_firing_rate(region_avg_dynamics_state, -i))
        r_1.append(first_order_difference(region_avg_dynamics_state, -i, -(i+1)))
    r_1 = np.mean(r_1)
    r = np.mean(r)
    if 0<r_1<0.05 or -0.03<r_1<0:
        return beta_r(r)
    else:
        return beta_first_order_difference(r_1)
    
beta = 15
top_down_time = 40000
no_top_down_time = 400000
save_inter_step = 40000

## top down
model.rebuild_up_param()
External_field_prior = copy.deepcopy(face_mask) * 2
initial_state = np.where(face_mask+object_mask==0, 1, -1)
stable_state = model.stochastic_dynamics([initial_state.reshape(-1)], beta=beta, 
                                         H_prior=External_field_prior, H_bottom_up=initial_state, 
                                         epochs=top_down_time, save_inter_step=save_inter_step)
plt.figure()
plt.imshow(stable_state.reshape(200,200));plt.colorbar();plt.axis('off')
Dynamic_states = model.dynamics_state
## dynamics only (posterior steps)
External_field_prior = np.zeros((200,200))
stable_state = model.stochastic_dynamics_changed_beta([stable_state.reshape(-1)], 
                                                      beta=beta, changed_beta_func=changed_beta_func, tao=1,
                                                      H_prior=External_field_prior, H_bottom_up=initial_state, 
                                                      epochs=no_top_down_time, save_inter_step=save_inter_step)
Dynamic_states = np.vstack((Dynamic_states, model.dynamics_state))

plt.figure(dpi=300)
face_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, face_mask)+1)/2
object_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, object_mask)+1)/2
body_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, limb_mask)+1)/2
place_mean_act = (model.avg_activation_in_mask_timeserise(Dynamic_states, place_mask)+1)/2
plt.plot(face_mean_act, label='Avg activation in face mask')
plt.plot(object_mean_act, label='Avg activation in object mask')
plt.plot(place_mean_act, label='Avg activation in place mask')
plt.plot(body_mean_act, label='Avg activation in body mask')
plt.legend()

plt.figure(dpi=300)
face_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, face_mask)
object_mean_act = model.avg_activation_in_mask_timeserise(model.Beta_maps, object_mask)
plt.plot(face_mean_act, label='Avg betas in face mask')
plt.plot(object_mean_act, label='Avg betas in object mask')
plt.legend()






