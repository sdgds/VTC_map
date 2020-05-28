#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from tqdm import tqdm
import copy
import numpy as np
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import networkx as nx
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/nfs/s2/userhome/zhangyiyuan/My_mac/Brain_SOM')
import BrainSOM


############################################
######### Train SOM with Hebb rule #########
############################################
SOM_act = np.load('/Users/mac/Desktop/TDCNN/SOM_act.npy')
Class_avg = []
for j in [i*50 for i in range(1000)]:
    Class_avg.append(SOM_act[j:j+50, :, :].mean(axis=0).reshape(-1))
Class_avg = np.array(Class_avg)


## Structure M with Q-Hebb
Q = np.zeros((1000,1000))
for i in range(1000):
    for j in range(1000):
        q = (1/4096)*np.dot(Class_avg[i].reshape(1,4096),Class_avg[j].reshape(4096,1))
        Q[i,j] = q
Q_inv = np.linalg.inv(Q)

M = (1.2/4096)*Class_avg.T.dot(Q_inv).dot(Class_avg)
M = M - 0.005
for diag in range(4096):
    M[diag, diag] = 0
plt.imshow(M,'flag');plt.colorbar()


## Structure M with E-I Hebb



# Prune some synapses
def prune_weights(M, alpha, u, sig):
    W = np.where(M>=0,
                 M-(M**alpha)*np.random.normal(u,sig),
                 -(np.abs(M)-(np.abs(M)**alpha)**np.random.normal(u,sig)))
    return W

for t in tqdm(range(10)):
    M = prune_weights(M, 0.3, 0.05, 0.05)
plt.imshow(M,'flag');plt.colorbar()


# Dynamic system for a special pattern
def dev_F(x):
    return np.where(x<=0, 0, 0.83)

attractor_pattern = SOM_act[0].reshape(-1)
temp = M.dot(attractor_pattern)
sign_h = dev_F(np.array([temp for i in range(4096)]).T)
F = (sign_h*M-np.eye(M.shape[0]))/1.2
eigvalue,eigvector = np.linalg.eig(F)
plt.figure()
plt.plot(eigvalue.real)
plt.figure()
plt.plot(eigvalue.imag)


# Weights distribution
plt.figure()
n,bins,patches = plt.hist(M.reshape(-1)[np.where(M.reshape(-1)>0)[0]], bins=100)
plt.figure(figsize=(10,7))
plt.scatter(np.log(np.abs(bins[:-1])), np.log(n))
plt.figure()
n,bins,patches = plt.hist(M.reshape(-1)[np.where(M.reshape(-1)<0)[0]], bins=100)
plt.figure(figsize=(10,7))
plt.scatter(np.log(np.abs(bins[:-1])), np.log(n))


## Graph
def Graph_position(index):
    coordinates = np.unravel_index(index, (64,64))
    vnode = zip(coordinates[0], coordinates[1])
    npos = dict(zip(index, vnode))
    return npos
    
G = nx.Graph()
M = np.where((M>np.percentile(M,99)) | (M<np.percentile(M,1)), M, 0)
index = np.random.choice(np.arange(0,4096), 3000)
for i in index:
    for j in index:
        G.add_edge(i,j,weight=M[i,j])

#正负权重二分图
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']>0]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight']<0]
pos = Graph_position(index)   #节点位置
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G,pos,edgelist=elarge,width=3,edge_color='r')
nx.draw_networkx_edges(G,pos,edgelist=esmall,alpha=0.5,edge_color='b',style='dashed')
plt.show()

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
edgewidth_positive = [d['weight']*5 for (u,v,d) in G.edges(data=True) if d['weight'] >0]
edgewidth_negative = [np.abs(d['weight'])*5 for (u,v,d) in G.edges(data=True) if d['weight'] <0]
pos = Graph_position(index)
plt.figure(figsize=(10,10)) 
nx.draw_networkx_nodes(G, pos, node_size=10, node_shape='.')
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=edgewidth_positive, edge_color='r')
nx.draw_networkx_edges(G, pos, edgelist=esmall, alpha=0.6, width=edgewidth_negative,
                       edge_color='b', style='dashed')

# Small world
C_G = nx.average_clustering(G)
L_G = nx.average_shortest_path_length(G)

ER = nx.random_graphs.erdos_renyi_graph(4096, 0.8)
WS = nx.random_graphs.watts_strogatz_graph(4096, 10, 0.3)

    
## ODE
def F(x):
    return np.where(x<=0, 0, 0.83*x)

def Hamilton(M, v):
    hamilton = v.reshape(1,4096).dot(M).dot(v.reshape(4096,1))
    return -0.5*hamilton.item()

Energy = []
def diff_equation(v, t, M, H, tao, noise_level):
    h = H[int(t)]
#    if t < 2:
#        h = H[0]
#    if 10<t<12:
#        h = H[11]
#    else:
#        h = np.zeros(H[0].shape[0])
    Energy.append(Hamilton(M, v))
    Change = F((np.dot(M,v)+h))
    dvdt = tao * (-v + Change)
    return dvdt

def diff_equation(v, t, M, H, tao, noise_level):
    h = H[int(t)]
    Energy.append(Hamilton(M, v))
    Change = F((np.dot(M,v)+h))
    dvdt = tao * (-v + Change)
    for index,neuron_delta in enumerate(dvdt):
        v_ = copy.deepcopy(v)
        v_[index] = v[index]+neuron_delta/0.1
        E_delta = Hamilton(M, v_) - Hamilton(M, v)
        if E_delta < 0:
            pass
        else:
            random_number = np.random.rand(1)
            if random_number<np.exp(-0.5*E_delta):
                pass
            else:
                dvdt[index] = 0
    return dvdt

# two stim 
t = np.linspace(0,20,200)
H1 = np.array([SOM_act[0].reshape(-1) for i in range(10)])
H2 = np.array([SOM_act[250].reshape(-1) for i in range(10)])
H = dict(zip(np.arange(0,21,1), np.vstack((H1,H2))))
solution = odeint(diff_equation, H[0], t[:-1], args=(M, H, 1.2, 0))

# one stim
t = np.linspace(0,10,100)
H1 = np.array([SOM_act[0].reshape(-1) for i in range(10)])
H = dict(zip(np.arange(0,11,1), H1))
solution = odeint(diff_equation, H[0], t[:-1], args=(M, H, 1.2, 0))


# plt solution
plt.figure()
for neuron in range(M.shape[0]):
    plt.plot(solution[:,neuron])

plt.figure()
neurons = np.random.choice(np.arange(M.shape[0]), 100)
for neuron in neurons:
    plt.plot(solution[:,neuron])

plt.imshow(SOM_act[0])
plt.imshow(SOM_act[250])
plt.imshow(solution[:100,:].mean(axis=0).reshape(64,64))
plt.imshow(solution[100:,:].mean(axis=0).reshape(64,64))

plt.figure()
plt.ion()     # 开启一个画图的窗口
for i in range(solution.shape[0]):
    plt.imshow(solution[i,:].reshape(64,64), 'jet')
    plt.title('This is time: %d' %i)
    plt.axis('off')
    plt.pause(0.000000000000000001)       # 停顿时间
plt.pause(0)   # 防止运行结束时闪退


# State space with PCA
pca = PCA(0.999)
solution_pca1 = pca.fit_transform(solution[:100,:])
pca = PCA(0.999)
solution_pca2 = pca.fit_transform(solution[100:,:])

fig = plt.figure()
x = solution_pca1[:,0]
y = solution_pca1[:,1]
z = solution_pca1[:,2]
plt.ion()
for i in range(100):
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:i], y[:i], z[:i])  
    ax.set_xlim(solution_pca1[:,0].min(), solution_pca1[:,0].max())
    ax.set_ylim(solution_pca1[:,1].min(), solution_pca1[:,1].max())
    ax.set_zlim(solution_pca1[:,2].min(), solution_pca1[:,2].max())
    ax.grid(False)
    plt.title('This is time: %d' %i)
    plt.pause(0.01)  
x = solution_pca2[:,0]
y = solution_pca2[:,1]
z = solution_pca2[:,2]
for i in range(100):
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:i], y[:i], z[:i])  
    ax.set_xlim(solution_pca2[:,0].min(), solution_pca2[:,0].max())
    ax.set_ylim(solution_pca2[:,1].min(), solution_pca2[:,1].max())
    ax.set_zlim(solution_pca2[:,2].min(), solution_pca2[:,2].max())
    ax.grid(False)
    plt.title('This is time: %d' %i)
    plt.pause(0.01)  
plt.pause(0) 



## Functional Map
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
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()
def Functional_map(class_name, topk):  
    f = os.listdir('/Users/mac/Data/fLoc_stim/' + class_name)
    Response = []
    Winners_map = np.zeros((64,64))
    for index,pic in enumerate(f):
        img = Image.open('/Users/mac/Data/fLoc_stim/'+class_name+'/'+pic).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0) 
        Response.append(1/som.activate(alexnet(picimg).data.numpy()))   # activate is the correlation of x and w, so inverse activate is represente higher activation as better match unit
        break
    Response = np.array(Response)
    return Response, Winners_map
    
topk = 0
Response_adult,Winners_adult = Functional_map('adult', topk=topk)
Response_child,Winners_child = Functional_map('child', topk=topk)

Response_house,Winners_house = Functional_map('house', topk=topk)
Response_corridor,Winners_corridor = Functional_map('corridor', topk=topk)

Response_word,Winners_word = Functional_map('word', topk=topk)
Response_word_avg = np.mean(Response_word, axis=0)

Response_limb,Winners_limb = Functional_map('limb', topk=topk)
Response_body,Winners_limb = Functional_map('body', topk=topk)

Response_car,Winners_car = Functional_map('car', topk=3)
Response_instrument,Winners_instrument = Functional_map('instrument', topk=topk)

Response_scramble,Winners_scrambled = Functional_map('scrambled', topk=topk)


def F(x):
    return np.where(x<=0, 0, 0.83*x)
def diff_equation(v, t, M, H, tao, noise_level):
    h = H[int(t)]
    Change = F((np.dot(M,v)+h)) + np.random.normal(0,noise_level,h.shape)
    dvdt = [tao*(-v[neuron]+Change[neuron]) for neuron in range(M.shape[0])]
    return dvdt

t = np.linspace(0,59,600)
H1 = np.array([Response_adult[0,:,:].reshape(-1) for i in range(10)])
H2 = np.array([Response_house[0,:,:].reshape(-1) for i in range(10)])
H3 = np.array([Response_word[0,:,:].reshape(-1) for i in range(10)])
H4 = np.array([Response_limb[0,:,:].reshape(-1) for i in range(10)])
H5 = np.array([Response_car[0,:,:].reshape(-1) for i in range(10)])
H6 = np.array([Response_scramble[0,:,:].reshape(-1) for i in range(10)])
H = dict(zip(np.arange(0,61,1), np.vstack((H1,H2,H3,H4,H5,H6))))
solution = odeint(diff_equation, H[0], t[:-1], args=(M, H, 1.2, 0))

plt.figure()
neurons = np.random.choice(np.arange(M.shape[0]), 100)
for neuron in neurons:
    plt.plot(solution[:,neuron])

plt.figure()
plt.ion()  
for i in range(solution.shape[0]):
    plt.imshow(solution[i,:].reshape(64,64), 'jet')
    plt.title('This is time: %d' %i)
    plt.axis('off')
    plt.pause(0.001)    
plt.pause(0) 

plt.figure()
plt.imshow(Response_adult[0,:,:]-Response_car[0,:,:], cmap='jet')
plt.figure()
plt.imshow(Response_house[0,:,:]-Response_car[0,:,:], cmap='jet')
plt.figure()
plt.imshow(Response_word[0,:,:]-Response_car[0,:,:], cmap='jet')
plt.figure()
plt.imshow(Response_limb[0,:,:]-Response_car[0,:,:], cmap='jet')
plt.figure()
plt.imshow(Response_car[0,:,:]-Response_scramble[0,:,:], cmap='jet')

plt.figure()
plt.imshow(solution[99,:].reshape(64,64)-solution[499,:].reshape(64,64), cmap='jet')
plt.figure()
plt.imshow(solution[199,:].reshape(64,64)-solution[499,:].reshape(64,64), cmap='jet')
plt.figure()
plt.imshow(solution[299,:].reshape(64,64)-solution[499,:].reshape(64,64), cmap='jet')
plt.figure()
plt.imshow(solution[399,:].reshape(64,64)-solution[499,:].reshape(64,64), cmap='jet')
plt.figure()
plt.imshow(solution[499,:].reshape(64,64)-solution[598,:].reshape(64,64), cmap='jet')


