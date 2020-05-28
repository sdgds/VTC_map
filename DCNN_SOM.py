import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('/nfs/s2/userhome/zhangyiyuan/My_mac/Brain_SOM')
import BrainSOM


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
        
alexnet = torchvision.models.alexnet()
alexnet.load_state_dict(torch.load('/nfs/s2/dnnbrain_data/models/alexnet.pth'))
alexnet.eval()

labels = dict()
f = open('/nfs/e3/ImgDatabase/ImageNet_2012/label/val.txt')
l = f.readlines()
for pair in l:
    labels[pair[:28]] = pair[29:-1]

Data = []
Validation = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/')
for v in Validation:
     if v[0]!='n':
         Validation.remove(v)
Data_label = np.array([])
for Classes in tqdm(Validation):
    piclist = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'+Classes)
    for pic in piclist:
        Data_label = np.append(Data_label, labels[pic])
np.save('/nfs/s2/userhome/zhangyiyuan/Desktop/DCNN_SOM/Data_label.npy', Data_label)

# Last layer
Data = []
Validation = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/')
for v in Validation:
     if v[0]!='n':
         Validation.remove(v)
for Classes in tqdm(Validation):
    piclist = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'+Classes)
    for pic in piclist:
        picdir = '/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'+Classes+'/'+pic
        img = Image.open(picdir).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0)
        Data.append(alexnet(picimg).data.numpy().reshape(-1))
Data = np.array(Data)
np.save('/nfs/s2/userhome/zhangyiyuan/Desktop/DCNN_SOM/Data.npy', Data)


# Conv5 out
Data = []
Validation = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/')
for v in Validation:
     if v[0]!='n':
         Validation.remove(v)
for Classes in tqdm(Validation):
    piclist = os.listdir('/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'+Classes)
    for pic in piclist:
        picdir = '/nfs/e3/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val/'+Classes+'/'+pic
        img = Image.open(picdir).convert('RGB')
        picimg = data_transforms['val'](img).unsqueeze(0)
        Data.append(alexnet.features(picimg).data.numpy().reshape(-1))
Data = np.array(Data)
np.save('/nfs/s2/userhome/zhangyiyuan/Desktop/DCNN_SOM/Data_Conv5.npy', Data)


# Conv5 relu out
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
        
Data = []
Validation = os.listdir('/home/dell/Desktop/Imgnet/valdata/')
for v in Validation:
     if v[0]!='I':
         Validation.remove(v)
alexnet_conv5_relu = NET(alexnet, 11)
for pic in tqdm(Validation):
    picdir = '/home/dell/Desktop/Imgnet/valdata/'+pic
    img = Image.open(picdir).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    alexnet_conv5_relu.layeract(picimg)
    Data.append(alexnet_conv5_relu.feature_map.data.numpy().mean((2,3)).reshape(-1))
Data = np.array(Data)
np.save('/home/dell/TDCNN/Data_Conv5_relu.npy', Data)


# Resnet 1000 out
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
Data = []
Validation = os.listdir('/home/dell/Desktop/Imgnet/valdata/')
for v in Validation:
     if v[0]!='I':
         Validation.remove(v)
for pic in tqdm(Validation):
    picdir = '/home/dell/Desktop/Imgnet/valdata/'+pic
    img = Image.open(picdir).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0).to(device)
    Data.append(resnet50(picimg).cpu().data.numpy().reshape(-1))
Data = np.array(Data)
np.save('/home/dell/TDCNN/Data_Resnet50.npy', Data)



#############################
######### Train SOM #########
#############################
# Based on last layer
def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))

def none_decay(scalar, t, max_iter):
    return scalar

som = BrainSOM.VTCSOM(64, 64, 1000, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
Data = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_fc8_SOM/Data.npy')
som.pca_weights_init(Data)
q_error, t_error = som.Train(Data, 50000, step_len=50000, verbose=False)
np.save('/nfs/s2/userhome/zhangyiyuan/Desktop/DCNN_SOM/Alexnet_fc8_SOM.npy', som._weights)

plt.figure()
plt.plot(q_error)
plt.ylabel('quantization error')
plt.xlabel('iteration index')
plt.show()

plt.figure()
plt.plot(t_error)
plt.ylabel('topographic error')
plt.xlabel('iteration index')
plt.show()


# Based on FC6
som = BrainSOM.VTCSOM(64, 64, 256, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
Data = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_conv5_SOM/Data_Conv5.npy')
Data_avg_channel = []
for i in range(Data.shape[0]):
    Data_avg_channel.append(Data[i].reshape(256,-1).mean(axis=1))
Data_avg_channel = np.array(Data_avg_channel)
som.pca_weights_init(Data_avg_channel)
q_error, t_error = som.Train(Data_avg_channel, 100000, step_len=100000, verbose=False)


# Based on conv5 relu out
som = BrainSOM.VTCSOM(64, 64, 256, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
Data = np.load('/Users/mac/Desktop/TDCNN/Results/Alexnet_conv5_SOM/Data_conv5_relu.npy')
som.pca_weights_init(Data)
q_error, t_error = som.Train(Data, 50000, step_len=50000, verbose=False)


# Based on Resnet50 last layer
som = BrainSOM.VTCSOM(64, 64, 1000, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
Data = np.load('/Users/mac/Desktop/TDCNN/Results/Resnet_1000_SOM/Data_Resnet50.npy')
som.pca_weights_init(Data)
q_error, t_error = som.Train(Data, 50000, step_len=50000, verbose=False)


