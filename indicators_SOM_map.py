#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import numpy as np
import torch
import scipy.stats as stats
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
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
som._weights = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc6_SOM/pca_init.npy')


# fLoc dataset
class NET(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.feature_map = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.feature_map = layer_out
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
alexnet_conv5_relu = NET(alexnet, 11)

def Functional_map(class_name, topk):  
    f = os.listdir('/Users/mac/Data/fLoc_stim/' + class_name)
    Response = []
    Winners_map = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    for index,pic in enumerate(f):
        img = Image.open('/Users/mac/Data/fLoc_stim/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        Response.append(1/som.activate(alexnet(picimg).data.numpy()))   # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
        #alexnet_conv5_relu.layeract(picimg)
        #Response.append(1/som.activate(alexnet_conv5_relu.feature_map.data.numpy().mean((2,3)).reshape(-1))) 
        for i in range(topk):
            Winners_map += som.activation_response(alexnet(picimg).data.numpy(), k=i)
            #Winners_map += som.activation_response(alexnet.features(picimg).data.numpy().mean((2,3)), k=i)
    Response = np.array(Response)
    return Response, Winners_map
    
topk = 0
Response_adult,Winners_adult = Functional_map('adult', topk=topk)
Response_adult_avg = np.mean(Response_adult, axis=0)
Response_child,Winners_child = Functional_map('child', topk=topk)
Response_child_avg = np.mean(Response_child, axis=0)
Response_face_avg = (Response_adult_avg + Response_child_avg)/2

Response_house,Winners_house = Functional_map('house', topk=topk)
Response_house_avg = np.mean(Response_house, axis=0)
Response_corridor,Winners_corridor = Functional_map('corridor', topk=topk)
Response_corridor_avg = np.mean(Response_corridor, axis=0)
Response_place_avg = (Response_house_avg + Response_corridor_avg)/2

Response_word,Winners_word = Functional_map('word', topk=topk)
Response_word_avg = np.mean(Response_word, axis=0)

Response_limb,Winners_limb = Functional_map('limb', topk=topk)
Response_limb_avg = np.mean(Response_limb, axis=0)
Response_body,Winners_limb = Functional_map('body', topk=topk)
Response_body_avg = np.mean(Response_body, axis=0)
Response_bodies_avg = (Response_limb_avg + Response_body_avg)/2

Response_car,Winners_car = Functional_map('car', topk=3)
Response_car_avg = np.mean(Response_car, axis=0)
Response_instrument,Winners_instrument = Functional_map('instrument', topk=topk)
Response_instrument_avg = np.mean(Response_instrument, axis=0)
Response_object_avg = (Response_car_avg + Response_instrument_avg)/2

Response_scramble,Winners_scrambled = Functional_map('scrambled', topk=topk)
Response_scramble_avg = np.mean(Response_scramble, axis=0)


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
Orignal_response = [(Response_adult + Response_child)/2,
                    (Response_house + Response_corridor)/2,
                    (Response_limb + Response_body)/2,
                    (Response_car + Response_instrument)/2]
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
    s1_ = (x1.shape[0]-1)*(s1**2)
    s2 = x2.std()
    s2_ = (x2.shape[0]-1)*(s2**2)
    s_within = np.sqrt((s1_+s2_)/(x1.shape[0]+x2.shape[0]-2))
    return (x1.mean()-x2.mean())/s_within

Orignal_response = [(Response_adult + Response_child)/2,
                    (Response_house + Response_corridor)/2,
                    (Response_limb + Response_body)/2,
                    (Response_car + Response_instrument)/2]

threshold_cohend = 2
    
t_map, p_map = stats.ttest_ind(Orignal_response[0], Orignal_response[3])
face_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Orignal_response[0][:,i,j], Orignal_response[3][:,i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            face_mask[i,j] = 1
print('Area', face_mask.sum())
plt.plot(np.sort(Cohend), label='face cohen d')

t_map, p_map = stats.ttest_ind(Orignal_response[1], Orignal_response[3])
place_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Orignal_response[1][:,i,j], Orignal_response[3][:,i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            place_mask[i,j] = 1
print('Area', place_mask.sum())
plt.plot(np.sort(Cohend), label='place cohen d')

t_map, p_map = stats.ttest_ind(Orignal_response[2], Orignal_response[3])
limb_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Orignal_response[2][:,i,j], Orignal_response[3][:,i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            limb_mask[i,j] = 1
print('Area', limb_mask.sum())
plt.plot(np.sort(Cohend), label='limb cohen d')

t_map, p_map = stats.ttest_ind(Orignal_response[3], Response_scramble)
object_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
Cohend = []
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        cohend = cohen_d(Orignal_response[3][:,i,j], Response_scramble[:,i,j])
        Cohend.append(cohend)
        if (p_map[i,j] < 0.01/4096) and (cohend>threshold_cohend):
            object_mask[i,j] = 1
print('Area', object_mask.sum())
plt.plot(np.sort(Cohend), label='object cohen d')
plt.plot(threshold_cohend*np.ones(len(Cohend)), label='threshold_cohend', linestyle='--')
plt.legend()

    
a_b_overlap(face_mask, place_mask)

plt.figure(figsize=(6,10))
plt.imshow(face_mask, cmap='Reds', alpha=1, label='face')
plt.imshow(place_mask, cmap='Greens',  alpha=0.4, label='place')
plt.imshow(limb_mask, cmap='Oranges',  alpha=0.3, label='limb')
plt.imshow(object_mask, cmap='Blues',  alpha=0.2, label='object')
plt.axis('off')



### MFS
# Face vs Place                
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
Orignal_response = [(Response_adult + Response_child)/2,
                    (Response_house + Response_corridor)/2,
                    (Response_limb + Response_body)/2,
                    (Response_car + Response_instrument)/2]
threshold_cohend = 0.5
plt.figure(figsize=(8,8))
plt.title("The threshold of Cohen's d: %s" %threshold_cohend)
for i in range(64):
    for j in range(64):
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
        if (p < 0.01/4096) and (cohend>threshold_cohend) and index in [0,1]:
            area_size[index] = area_size[index] + 1
            plt.scatter(i, j, color=color_map[index][0])

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
        response.append(1/som.activate(alexnet(picimg).data.numpy())) 
        #response.append(1/som.activate(alexnet.features(picimg).data.numpy().mean((2,3))))
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
        #response.append(1/som.activate(alexnet(picimg).data.numpy())) 
        response.append(1/som.activate(alexnet.features(picimg).data.numpy().mean((2,3))))
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
        #response.append(1/som.activate(alexnet(picimg).data.numpy())) 
        response.append(1/som.activate(alexnet.features(picimg).data.numpy().mean((2,3))))    
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

color_map = {0:['red','face'],
             1:['green','place'],
             2:['gold','body'],
             3:['midnightblue','object']}
Orignal_response = [(Response_adult + Response_child)/2,
                    (Response_house + Response_corridor)/2,
                    (Response_limb + Response_body)/2,
                    (Response_car + Response_instrument)/2]
for i in range(64):
    for j in range(64):
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
        if (p < 0.01/4096) and (cohend>threshold_cohend) and index==0:
            plt.scatter(i, j, color='black', marker='x', alpha=0.8)
        if (p < 0.01/4096) and (cohend>threshold_cohend) and index==2:
            plt.scatter(i, j, color='black', marker='.', alpha=0.8)



###############################################################################
###############################################################################
### Why can Alexnet+SOM can generate MFS, Nested, and Overlap
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
    
## MFS
# Face vs Place          
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
def Alexnet_fLoc(class_name):  
    f = os.listdir('/Users/mac/Data/fLoc_stim/' + class_name)
    Response = []
    for index,pic in enumerate(f):
        img = Image.open('/Users/mac/Data/fLoc_stim/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        Response.append(alexnet(picimg).data.numpy().reshape(-1))  # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
        #Response.append(alexnet.features(picimg).data.numpy().mean((2,3))) 
    Response = np.array(Response)
    return Response
       

Response_adult = Alexnet_fLoc('adult')
Response_child = Alexnet_fLoc('child')

Response_house = Alexnet_fLoc('house')
Response_corridor = Alexnet_fLoc('corridor')

Response_limb = Alexnet_fLoc('limb')
Response_body = Alexnet_fLoc('body')

Response_car = Alexnet_fLoc('car')
Response_instrument = Alexnet_fLoc('instrument')


Orignal_response = [(Response_adult + Response_child)/2,
                    (Response_house + Response_corridor)/2,
                    (Response_limb + Response_body)/2,
                    (Response_car + Response_instrument)/2]

Final_response = [Orignal_response[0]-(Orignal_response[1]+Orignal_response[2]+Orignal_response[3])/3,
                  Orignal_response[1]-(Orignal_response[0]+Orignal_response[2]+Orignal_response[3])/3,
                  Orignal_response[2]-(Orignal_response[1]+Orignal_response[0]+Orignal_response[3])/3,
                  Orignal_response[3]-(Orignal_response[1]+Orignal_response[2]+Orignal_response[0])/3]

Category = np.vstack((Final_response[0],Final_response[1],Final_response[2],Final_response[3]))
plt.figure()
plt.imshow(np.corrcoef(Category[:288]), cmap='jet')
plt.axis('off');plt.colorbar()


# Animate vs Inanimate
def Alexnet_caltech(picdir):
    f = os.listdir(picdir)
    for i in f:
        if i[-3:] != 'jpg':
            f.remove(i)
    response = []
    for pic in f:
        img = Image.open(picdir+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        response.append(alexnet(picimg).data.numpy().reshape(-1))
        #response.append(alexnet.features(picimg).data.numpy().mean((2,3)))
    return np.array(response)

Animate_response = Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/253.faces-easy-101/')
Animate_response = np.vstack((Animate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/159.people/')))
Animate_response = np.vstack((Animate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/060.duck/')))
Animate_response = np.vstack((Animate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/056.dog/')))
Animate_response = np.vstack((Animate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/087.goldfish/')))

Inanimate_response = Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/091.grand-piano-101/')
Inanimate_response = np.vstack((Inanimate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/136.mandolin/')))
Inanimate_response = np.vstack((Inanimate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/146.mountain-bike/')))
Inanimate_response = np.vstack((Inanimate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/172.revolver-101/')))
Inanimate_response = np.vstack((Inanimate_response, Alexnet_caltech('/Users/mac/Data/caltech256/256_ObjectCategories/252.car-side-101/')))

plt.figure()
plt.imshow(np.corrcoef(np.vstack((Animate_response-Inanimate_response.mean(axis=0),
                                  Inanimate_response-Animate_response.mean(axis=0)))), 
           cmap='jet')
plt.axis('off');plt.colorbar()


### Nested spacial
plt.figure()
plt.imshow(np.corrcoef(np.vstack((Animate_response-Inanimate_response.mean(axis=0),
                                 Category[0:144],
                                 Category[288:432],
                                 Category[144:288],
                                 Category[432:]))), cmap='jet')           
plt.axis('off');plt.colorbar()         
            

