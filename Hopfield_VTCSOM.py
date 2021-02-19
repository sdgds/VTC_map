#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
import warnings
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
import BrainSOM
import dhnn


### Data
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
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som
    

class Stochastic_Hopfield_nn(dhnn.DHNN):
    def __init__(self, x, y, pflag, nflag, right_pattern):
        self._x = x
        self._y = y
        self._neigx = np.arange(x)
        self._neigy = np.arange(y) 
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        self.N = x*y
        self.pflag = pflag
        self.nflag = nflag
        self.state = 0
        self.right = right_pattern
    
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T
    
    def reconstruct_w(self, data):
        """Training pipeline.
        Arguments:
            data {list} -- each sample is vector
        Keyword Arguments:
            issave {bool} -- save weight or not (default: {True})
            wpath {str} -- the local weight path (default: {'weigh.npy'})
        """
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        #self._w = (1/self.N) * (1/mat.shape[0]) * self._w
        self._w = (1/self.N) * self._w
            
    def reconstruct_w_with_structure_constrain(self, data, sigma):
        # Hebb weights
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        # Gaussian structure
        Gamma_matrix = np.zeros((self._x*self._y, self._x*self._y))
        for i in tqdm(range(self._x)):
            for j in range(self._y):
                index = i*self._x + j
                temp = self._gaussian((i,j),sigma).reshape(-1)
                Gamma_matrix[index] = np.where(temp>=0.01, 1, 0)
        # new weights
        self._w = self._w * Gamma_matrix
        self._w = (1/self.N) * self._w
        del Gamma_matrix
    
    def stochastic_activation(self, beta, bi):
        return 1/(1+np.exp(-beta*bi))
    
    def H_liklihood(self, state, right):
        order = np.mean(state*right)
        temp = (1/order) - 1
        H = np.ones((self._x*self._y), 1)
        H = H * temp
        return H
    
    def stochastic_dynamics(self, state, beta, H_prior, H_lik=False, epochs=1000):
        """
        beta: noise
        H_prior: external field from higher area, the size is (200x200)
        """
        if isinstance(state, list):
            state = np.asarray(state)
            print(state.shape)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        for ind in tqdm(indexs):
            if H_lik==False:
                diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_prior))
            if H_lik==True:
                diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_prior + self.H_liklihood(state.reshape(self._x,self._y), self.right)))
            diagonal = np.expand_dims(diagonal, -1)
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
        self.state = state.reshape(self._x,self._y)
        return state   
    
    def order_parameter(self, state, right):
        return np.mean(state * right)
    
    
def plot_memory(initial_state, state, memory_pattern):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.imshow(initial_state,cmap='jet')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(state,cmap='jet')
    plt.colorbar() 
    plt.subplot(133)
    plt.imshow(memory_pattern,cmap='jet')
    plt.colorbar() 
    



### sigma=6.2
som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian', random_seed=0)
som._weights = np.load('D:\\TDCNN\\Results\\Alexnet_fc8_SOM\\SOM_norm(200x200)_pca4_Sigma_200000step\som_sigma_3.4.npy')

Data = np.load('D:\\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
Data = zscore(Data)
pca = PCA()
pca.fit(Data)
Response_som = Functional_map_pca(som, pca, [0,1,2,3])
threshold = np.percentile(Response_som,97,axis=(1,2))
Response_som_truncate = np.where(Response_som-threshold.reshape(360,1,1)>0,1,-1)

Response_som_avg = np.vstack((Response_som[:111,:,:].mean(axis=0,keepdims=True),
                              Response_som[111:172,:,:].mean(axis=0,keepdims=True),
                              Response_som[172:250,:,:].mean(axis=0,keepdims=True),
                              Response_som[250:,:,:].mean(axis=0,keepdims=True)))
Response_som_avg_truncate = np.vstack((np.where(Response_som_avg[0]>np.percentile(Response_som_avg[0],80),1,-1).reshape(1,200,200),
                                        np.where(Response_som_avg[1]>np.percentile(Response_som_avg[1],80),1,-1).reshape(1,200,200),
                                        np.where(Response_som_avg[2]>np.percentile(Response_som_avg[2],90),1,-1).reshape(1,200,200),
                                        np.where(Response_som_avg[3]>np.percentile(Response_som_avg[3],70),1,-1).reshape(1,200,200)))
Response_som_avg_truncate_flat = Response_som_avg_truncate.reshape(4,-1)



model = Stochastic_Hopfield_nn(x=200, y=200, pflag=1, nflag=-1,
                               right_pattern=Response_som_avg_truncate_flat[0])
model.reconstruct_w([Response_som_avg_truncate_flat])
#model.reconstruct_w([Response_som_truncate.reshape(360,-1)])

External_field_prior = np.zeros((200,200))
#External_field_prior[50:75,50:75] = 10
Delta_face = []
Delta_place = []
for beta in np.arange(0, 5.1, 0.5):
    recovery = model.stochastic_dynamics([Response_som_truncate[0].reshape(-1)], 
                                          beta=beta, H_prior=External_field_prior, 
                                          H_lik=False, epochs=50000)
    plot_memory(Response_som_truncate[0], 
                recovery[0].reshape(200,200),
                Response_som_avg_truncate_flat[0].reshape(200,200))
    Delta_face.append(model.order_parameter(recovery[0], Response_som_avg_truncate_flat[0]))
    Delta_place.append(model.order_parameter(recovery[0], Response_som_avg_truncate_flat[1]))

plt.figure()
plt.plot(Delta_face)
plt.plot(Delta_place)


