# -*- coding: utf-8 -*-
import os
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import zscore


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

def sigmoid(x):
    return 1/(1+np.exp(-0.01*(x-450)))

def pic_size(pos_index):
    """
    pos_index is image position
    return: the size of image
    """
    d = np.linalg.norm(pos_index)
    d = sigmoid(d) + 1
    d = np.int0(200*d)
    return (d-50,d)

def plant_pic_in_3D(IMGs,Is,Js,Ks,save_dir):
    """
    IMGs is list with many images(array)
    Is, Js, Ks are corrdinate of IMGs
    """
    A,B,C,a,b,c = Is.max(),Js.max(),Ks.max(),Is.min(),Js.min(),Ks.min()
    canvas_size = 700
    Is += canvas_size
    Ks += canvas_size
    plt.style.use('default')
    fig = plt.figure(dpi=500)
    ax = fig.gca(projection='3d')
    # axis
    x = np.linspace(-canvas_size, canvas_size, canvas_size*2)
    y = np.linspace(canvas_size, -canvas_size, canvas_size*2)
    X,Y = np.meshgrid(x, y)
    for i in tqdm(range(Is.shape[0])):
        # pic
        img = IMGs[i]
        size = img.shape[:2]
        picimg = np.zeros((canvas_size*2,canvas_size*2,4))
        picimg[int(Is[i]-size[0]/2):int(Is[i]+size[0]/2),int(Ks[i]-size[1]/2):int(Ks[i]+size[1]/2),:] = img
        # plot
        ax.plot_surface(X, Y=Js[i], Z=Y, rstride=1, cstride=1, facecolors=picimg, shade=False) 
    ax.set_aspect("auto")
    ax.set_xlim(a-100, A+100)
    ax.set_ylim(b-100, B+100)
    ax.set_zlim(c-100, C+100)
    plt.axis('off')
    plt.savefig(save_dir, format="png", dpi=500)
    
        
Data = np.load('D:\TDCNN\Results\Alexnet_fc8_SOM\Data.npy')
pca = PCA()
pca.fit(Data)

Validation = os.listdir('D:\\Brain_data\\valdata')
Validation = np.random.choice(Validation, 100)
Data_100_pca = []
IMGs = []
for pic in tqdm(Validation):
    picdir = 'D:\\Brain_data\\valdata\\'+pic
    img = Image.open(picdir).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    out = alexnet(picimg).data.numpy().reshape(-1)
    pos_index = pca.transform(out.reshape(1,-1))[0,[0,1,2]]
    Data_100_pca.append(pos_index)
    IMGs.append(np.array(Image.open(picdir).convert('RGBA').resize((60,120)))/255)
Data_100_pca = np.array(Data_100_pca)  
Data_100_pca = 200 * zscore(Data_100_pca, axis=0)  

plant_pic_in_3D(IMGs,
                Data_100_pca[:,0],
                Data_100_pca[:,1],
                Data_100_pca[:,2],
                'C:/Users/12499/Desktop/object_space.png')

   
        