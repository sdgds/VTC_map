#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4/16/2020
BrainSOM mapping AI to HumanBrain 
@author: Zhangyiyuan
"""

import h5py
import brian2 as b2
from neurodynex3.leaky_integrate_and_fire import LIF
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.adex_model import AdEx
import PIL.Image as Image
import sys
from tqdm import tqdm
from time import time
from datetime import timedelta
import numpy as np
from scipy.ndimage import zoom
from scipy.misc import logsumexp
from scipy.integrate import odeint
from scipy.stats import zscore
from warnings import warn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import minisom
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow.compat.v1 as tf
from multiprocessing.dummy import Pool as ThreadPool
import sys
sys.path.append('/Users/mac/Desktop/My_module')
import Guided_CAM
import CAM_lastFC



def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations

def _wrap_index__in_verbose(iterations):
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    sys.stdout.write(progress)
    beginning = time()
    sys.stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        sys.stdout.write(progress)

def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))

def asymptotic_decay(scalar, t, max_iter):
    return scalar / (1+t/(max_iter/2))

def none_decay(scalar, t, max_iter):
    return scalar




class VTCSOM(minisom.MiniSom):
    #### Initialization ####
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 sigma_decay_function=asymptotic_decay, lr_decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', random_seed=None):
        """
        x : int
            x dimension of the feature map.
        y : int
            y dimension of the feature map.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float
            Spread of the neighborhood function (sigma(t) = sigma / (1 + t/T) where T is num_iteration/2)
        learning_rate : 
            initial learning rate (learning_rate(t) = learning_rate / (1 + t/T)
        neighborhood_function : function, optional (default='gaussian')
            possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
        random_seed : int
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)
        self.theta = 0
        
        self.M = np.random.randn(4096,4096)
        self.Normalize_M()
        self.M = np.random.rand(4096,4096)
        self.Normalize_M()
        
        G = nx.random_graphs.watts_strogatz_graph(4096,10,0.3)
        self.M = np.zeros((4096,4096))
        for (u,v,d) in tqdm(G.edges(data=True)):
            self.M[u,v] = np.random.randn()
        self.Normalize_M()
            
        G = nx.random_graphs.barabasi_albert_graph(4096,5)
        self.M = np.zeros((4096,4096))
        for (u,v,d) in tqdm(G.edges(data=True)):
            self.M[u,v] = np.random.randn()
        self.Normalize_M()

        self._x = x
        self._y = y
        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function
        self._lr_decay_function = lr_decay_function
        self._sigma_decay_function = sigma_decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'circulate_gaussian': self._circulate_gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle,
                          'nowinner_gaussian': self._nowinner_gaussian}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]
                
    def Normalize_X(self, x):
        temp = np.sum(np.multiply(x, x))
        x /= np.sqrt(temp)
        return x
    
    def Normalize_W(self):
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)
        
    def Normalize_M(self):
        self.M /= np.linalg.norm(self.M, keepdims=True)
    
    def Retina_spontaneous_weights_init(self, Data):
        """random input as retina spontaneous, Alexnet is non-pretrained"""
        alexnet = torchvision.models.alexnet(pretrained=True)
        alexnet.eval()
        retina_data = torch.Tensor(np.random.uniform(0,1,(train_step_len,3,224,224)))
        inputs = alexnet(retina_data).data.numpy()
        self.Train(inputs, inputs.shape[0], step_len=inputs.shape[0], verbose=False)
        
    def Eccentricity_weights_init(self):
        """
        1. Human gaze
        2. channel wise attention like human
        3. init weights
        """
        def gaussian_weights(self, r, sigma):
            d = 2*np.pi*sigma*sigma
            ax = np.ones(64)
            ay = np.exp(-np.power(self._neigy-(32-r), 2)/d) + np.exp(-np.power(self._neigy-(32+r), 2)/d)
            return np.outer(ax, ay) 
        def amplification(data):
            return (1/(1+np.exp(-10*data))-0.5)*2
        
        data_transforms = {
            'attention': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])])
            }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        alexnet = torchvision.models.alexnet(pretrained=True).to(device)
        alexnet.eval()
        
        # Human gaze
        img = Image.open('/Users/mac/Desktop/My_module/deep_gaze/timg.jpeg')
        img = np.asarray(img)
        centerbias_template = np.load('/Users/mac/Desktop/My_module/deep_gaze/centerbias.npy')  
        centerbias = zoom(centerbias_template, (img.shape[0]/1024, img.shape[1]/1024), order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)
        
        image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
        centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]  # BHWC, 1 channel (log density)
        
        tf.reset_default_graph()
        tf.disable_eager_execution()
        
        check_point = '/Users/mac/Desktop/My_module/deep_gaze/DeepGazeII.ckpt'  # DeepGaze II
        new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
        
        input_tensor = tf.get_collection('input_tensor')[0]
        centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        log_density = tf.get_collection('log_density')[0]
        log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]
        
        with tf.Session() as sess:  
            new_saver.restore(sess, check_point)    
            log_density_prediction = sess.run(log_density, {
                input_tensor: image_data,
                centerbias_tensor: centerbias_data,
            })
            
        attention = log_density_prediction[0, :, :, 0]
        attention_img = Image.fromarray(np.uint8(255*(attention-attention.min())/(attention.max()-attention.min())))
        attention_img = data_transforms['attention'](attention_img)
        attention_im = np.asarray(attention_img)
        
        x_gaze = np.where(attention_im>=np.percentile(attention_im,95))[0].mean()
        y_gaze = np.where(attention_im>=np.percentile(attention_im,95))[1].mean()
        
        # channel wise attention
        alexnet = torchvision.models.alexnet(pretrained=True)
        alexnet.eval()
        
        img = Image.open('/Users/mac/Desktop/My_module/deep_gaze/timg.jpeg')
        img = data_transforms['val'](img).unsqueeze(0)
        x = img
        for layer_num in range(12):
            x = alexnet.features[layer_num](x)
        W = [] 
        for channel in tqdm(range(256)):
            channel_im = np.uint8(Image.fromarray(x[0,channel,:,:].data.numpy()).resize((224,224), Image.ANTIALIAS))
            if channel_im.any()==0:
                weight = 0
            else:
                weight = np.corrcoef(np.vstack((attention_im.reshape(-1), 
                                                channel_im.reshape(-1))))[0,1]
                if weight<0:
                    weight = 0
            x[0,channel,:,:] = weight * x[0,channel,:,:]
            W.append(weight)
        x = alexnet.features[12](x)
        x = alexnet.avgpool(x).view(-1)
        out = alexnet.classifier(x)
        
        for channel in tqdm(range(256)):
            alexnet.features[10].weight[:,channel,:,:] = alexnet.features[10].weight[:,channel,:,:] * W[channel]
            alexnet.features[10].bias[channel] = alexnet.features[10].bias[channel] * W[channel]
        alexnet_w = alexnet.state_dict()
        alexnet = torchvision.models.alexnet(pretrained=False)
        alexnet.load_state_dict(alexnet_w)
        alexnet.eval()
        
        # init weights
        self._weights = np.zeros(self._weights.shape)
        gcv2 = Guided_CAM.GradCam(alexnet, target_layer=12)
        img = img.to(device)
        def neuron_weights(neuron):
            cam,_ = gcv2.generate_cam(img, target_class=neuron)
            x = np.where(cam>=np.percentile(cam,95))[0].mean()
            y = np.where(cam>=np.percentile(cam,95))[1].mean()
            r = 32 * amplification(np.sqrt((x-x_gaze)**2 + (y-y_gaze)**2)/np.sqrt(224**2+224**2))
            r = round(r)
            self._weights[:,:,neuron] = gaussian_weights(self, r, sigma=5)
        pool = ThreadPool()
        pool.map(neuron_weights, range(1000))
        pool.close()
        pool.join()
        self.Normalize_W()
        
        
    
    def _circulate_gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        map_x = self._weights.shape[0]
        map_y = self._weights.shape[1]
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self._neigx-c[0], 2)/d) + np.exp(-np.power(self._neigx+(map_y-c[0]), 2)/d)
        ay = np.exp(-np.power(self._neigy-c[1], 2)/d) + np.exp(-np.power(self._neigy+(map_x-c[1]), 2)/d)
        return np.outer(ax, ay)  # the external product gives a matrix
    
    def _nowinner_gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*np.pi*sigma*sigma
        ax=0
        for i in range(self._weights.shape[0]):
            ax += np.exp(-np.power(self._neigx-i, 2)/d)
        ay=0
        for j in range(self._weights.shape[1]):
            ay += np.exp(-np.power(self._neigy-j, 2)/d)
        return np.outer(ax, ay)  # the external product gives a matrix   
    
    
    
    #### Training ####      
    ###########################################################################
    ###########################################################################
    def Train(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """            
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, end_num) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_forward(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """            
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            self.update(data[iteration], 
                        self.forward_winner(data[iteration]), 
                        t, num_iteration) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_multiwinner(self, data, num_iteration, step_len, verbose):      
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            self.multiwinner_update(data[iteration], 
                                    self.winner(data[iteration], k=0),
                                    self.winner(data[iteration], k=np.int(np.random.gamma(100,2))),
                                    t, num_iteration) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_with_eccentricity(self, data, num_iteration, step_len, verbose):
        def minmax(x):
            return (x-x.min())/(x.max()-x.min())
        def gaussian_weights(self, r, sigma):
            d = 2*np.pi*sigma*sigma
            ax = np.ones(64)
            ay = np.exp(-np.power(self._neigy-(32-r), 2)/d) + np.exp(-np.power(self._neigy-(32+r), 2)/d)
            return np.outer(ax, ay)
        def magnification_factor(x):
            lamda = 1
            x0 = 0.1
            temp = lamda * np.log(1+x/x0)
            return temp/2.3978952727983707
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        alexnet = torchvision.models.alexnet(pretrained=True).to(device)
        alexnet.eval()
        
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            img = data_transforms['val'](img).unsqueeze(0)
            img = img.to(device)
            # gaze point
            gcv = CAM_lastFC.GradCam(alexnet, target_layer=12)
            attention_im,_ = gcv.generate_cam(img)
            x_gaze = np.where(attention_im>=np.percentile(attention_im,99))[0].mean()
            y_gaze = np.where(attention_im>=np.percentile(attention_im,99))[1].mean()
            # ditinct fovea and hephical neuron
            t1=time()
            gcv2 = Guided_CAM.GradCam(alexnet, target_layer=12)
            Activation_filter = []
            R1 = []
            R2 = []
            def neuron_filter(neuron):
                cam,_ = gcv2.generate_cam(img, target_class=neuron)
                x = np.where(cam>=np.percentile(cam,99))[0].mean()
                y = np.where(cam>=np.percentile(cam,99))[1].mean()
                r = np.sqrt((x-x_gaze)**2 + (y-y_gaze)**2)/np.sqrt(224**2+224**2)
                R1.append(r)
                r = magnification_factor(r)
                R2.append(r)
                r = round(32 * r)
                activation_filter = gaussian_weights(self, r, sigma=5)
                activation_filter = activation_filter/activation_filter.max()
                Activation_filter.append(activation_filter)
            pool = ThreadPool()
            pool.map(neuron_filter, range(1000))
            pool.close()
            pool.join()
            Activation_filter = np.array(Activation_filter)
            print(time()-t1)
            # activations through activation_filter
            x = data[iteration]
            x = self.Normalize_X(x)
            S = []
            for neuron in range(1000):
                s = np.subtract(x[neuron], self._weights[:,:,neuron]) 
                s = (1-Activation_filter[neuron,:,:]) * s
                S.append(s)
            S = np.array(S)
            Activation_filted = minmax(torch.Tensor(np.linalg.norm(S, axis=0)))
            winner_x,winner_y = np.unravel_index(torch.multinomial(1-Activation_filted.reshape(-1),10),
                                 Activation_filted.shape)
            winner_neuron = (winner_x.mean(), winner_y.mean())
            # update weights
            self.update(data[iteration], 
                        winner_neuron, 
                        t, end_num) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_with_hebb_based_on_firing_rate_model(self, data, num_iteration, step_len, 
                                                   verbose, hebb_type='basic_hebb'):
        """
        1. Feedforward: Alexnet+SOM for all time in unstabel state
        2. Reccurent: firing model to stabel state
        3. Update W: SOM rule to update SOM weights, Hebb rule to update reccurent weights
        """
        ### training
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            ## all-time feedforward + reccurent dynamic system
            solution = self._activate_recurrent_firing_rate(data[iteration])
            
            ## update weights of W and M
            # update SOM weights
#            if som_type=='normal':
#                winner_neuron = np.unravel_index(np.argmin(solution[-1,:]), (self._x,self._y))
#                self.update(data[iteration], winner_neuron, t, end_num) 
#            if som_type=='None':
#                pass
            # update reccurent connection weights
            if hebb_type=='basic_hebb':
                self.basic_hebb_update(data[iteration], solution[-1,:], update_object='W')
                self.basic_hebb_update(data[iteration], solution[-1,:], update_object='M')
                self.Normalize_W()
                self.Normalize_M()
            if hebb_type=='oja_hebb_update':
                self.oja_hebb_update(data[iteration], solution[-1,:], gamma=0.5,
                                     update_object='W')
                self.oja_hebb_update(data[iteration], solution[-1,:], gamma=0.5,
                                     update_object='M')
                self.Normalize_W()
                self.Normalize_M()
            if hebb_type=='BCM_hebb_update':
                self.BCM_hebb_update(data[iteration], solution[-1,:], 
                                     tao_M=1, tao_theta=0.1, update_object='W')
                self.BCM_hebb_update(data[iteration], solution[-1,:], 
                                     tao_M=1, tao_theta=0.1, update_object='M')
                self.Normalize_W()
                self.Normalize_M()
            
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
    
    def Train_with_hebb_based_on_LIF(self, data, num_iteration, step_len, 
                                     verbose, hebb_type='basic_hebb'):
        """
        1. feedforward input by firing rate poisson spike train
        2. LIF output by feedforward
        3. recurrent connection
        4. recurrent+feedforward as input, LIF output
        5. training (SOM train and lateral conncection train)
        """
        def poisson_generator(n_neuron, rate, t):
            return np.random.poisson(rate, (n_neuron,t))
        
        ### training
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        q_error = np.array([])
        t_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num 
            # feedforward current
            Feed_current = np.zeros((self._x,self._y,100))
            self._activate(data[iteration])
            firing_rate = self._activation_map/self._activation_map.max()
            for x in range(self._x):
                for y in range(self._y):
                    rate = firing_rate[x,y]
                    feed_current = poisson_generator(10, rate, 100).sum(axis=0)
                    Feed_current[x,y,:] = feed_current
            # feedforward output
            Output_current = np.zeros((self._x,self._y,100))
            for x in tqdm(range(self._x)):
                for y in range(self._y):     
                    input_current = Feed_current[x,y,:].reshape(-1,1)
                    input_current = b2.TimedArray(input_current*b2.namp, dt=1.0*b2.ms)
                    state_monitor, spike_monitor = LIF.simulate_LIF_neuron(input_current,
                                                       simulation_time = 100 * b2.ms,
                                                       firing_threshold = -50 * b2.mV,
                                                       membrane_resistance = 10 * b2.Mohm,
                                                       membrane_time_scale = 8 * b2.ms,
                                                       abs_refractory_period = 2.0 * b2.ms)
                    output_spikes = np.zeros((100,1))
                    temp = np.array(spike_monitor.spike_trains()[0])*1000
                    temp = np.around(temp)
                    if temp[-1]==100:
                        temp[-1] -= 1
                    output_spikes[np.int0(temp)] = 1
                    Output_current[x,y,:] = output_spikes.reshape(-1)   
            # recurrent current
            Recurrent_current = np.zeros((self._x,self._y,100))
            for unit in range(self._x*self._y):
                recurrent_current = np.dot(self.M[unit,:].reshape(1,-1), Output_current.reshape(-1,100))
                Recurrent_current[x,y,:] = recurrent_current
            # all Input current and run LIF
            Input_current = Feed_current + Recurrent_current
            Output_current = np.zeros((self._x,self._y,100))
            for x in range(self._x):
                for y in range(self._y):     
                    input_current = Input_current[x,y,:].reshape(-1,1)
                    input_current = b2.TimedArray(input_current*b2.namp, dt=1.0*b2.ms)
                    state_monitor, spike_monitor = LIF.simulate_LIF_neuron(input_current,
                                                       simulation_time = 100 * b2.ms,
                                                       firing_threshold = -50 * b2.mV,
                                                       membrane_resistance = 10 * b2.Mohm,
                                                       membrane_time_scale = 8 * b2.ms,
                                                       abs_refractory_period = 2.0 * b2.ms)
                    output_spikes = np.zeros((100,1))
                    temp = np.array(spike_monitor.spike_trains()[0])*1000
                    temp = np.around(temp)
                    if temp[-1]==100:
                        temp[-1] -= 1
                    output_spikes[np.int0(temp)] = 1
                    Output_current[x,y,:] = output_spikes.reshape(-1)
            ### Update
            # update W
            winner_unit = np.unravel_index(Output_current.sum(axis=2).argmax(), (self._x, self._y))
            self.update(data[iteration], winner_unit, t, end_num) 
            # update M
            self.STDP_update(Output_current, tao=10, a=1, b=1)
            self.Normalize_W()
            self.Normalize_M()           
                    
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, self.quantization_error(data))
                t_error = np.append(t_error, self.topographic_error(data))
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data)) 
        return q_error, t_error
        
    
    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        s = np.subtract(x, self._weights)  # x - w
        self._activation_map = np.linalg.norm(s, axis=-1)
        
    def _forward_activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        s = np.dot(self._weights, x)
        self._activation_map = s
        
    def _activate_recurrent_firing_rate(self, x):
        def F(x):
            return np.where(x<=0, 0, 0.8*x)
        def diff_equation(v, t, M, H, tao, noise_level):
            h = H
            Change = F((np.dot(M,v)+h))
            dvdt = tao * (-v + Change)
            return dvdt
        ## all-time feedforward + reccurent dynamic system
        times = np.linspace(0,10,100)
        SOM_act = self.activate(x)
        H = 1/SOM_act.reshape(-1)
        solution = odeint(diff_equation, H, times, args=(self.M, H, 1.2, 0))
        return solution

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map
    
    def forward_activate(self, x):
        """Returns the activation map to x."""
        self._forward_activate(x)
        return self._activation_map
    
    def winner(self, x, k=0):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[k],
                                self._activation_map.shape)
        
    def forward_winner(self, x, k=0):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._forward_activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[-1],
                                self._activation_map.shape)
        
    def activation_response(self, data, k=0):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x, k)] += 1
        return a
    
    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._lr_decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        self.Normalize_W()
        
    def multiwinner_update(self, x, win1, win2, t, max_iteration):
        ## Winner neuron
        eta = self._lr_decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win1, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        
        ## Second Winner neuron
        factor = np.sort(self._activation_map.reshape(-1))[0]/np.sort(self._activation_map.reshape(-1))[1]
        eta = self._lr_decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._sigma_decay_function(self._sigma, t, max_iteration) * factor
        # improves the performances
        g = self.neighborhood(win2, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        
    def basic_hebb_update(self, x, H, update_object):
        h = H.reshape(-1,1)
        h /= np.sqrt(np.sum(np.multiply(h, h)))
        h = zscore(h)
        if update_object=='M':
            self.M += np.dot(h, h.T)
        if update_object=='W':
            for i in range(x.shape[0]):
                self._weights[:,:,i] += x[i]*h.reshape(self._x, self._y)
        
    def oja_hebb_update(self, x, H, gamma, update_object):
        h = H.reshape(-1,1)
        h /= np.sqrt(np.sum(np.multiply(h, h)))
        h = zscore(h)
        if update_object=='M':
            self.M += gamma*(np.dot(h,h.T)-self.M*h*h)
        if update_object=='W':
            h = h.reshape(self._x, self._y)
            for i in range(x.shape[0]):
                self._weights[:,:,i] += gamma*(x[i]*h-self._weights[:,:,i]*h*h)            
        
    def BCM_hebb_update(self, x, H, tao_M, tao_theta, update_object):
        """Synapse competition"""
        h = H.reshape(-1,1)
        h /= np.sqrt(np.sum(np.multiply(h, h)))
        h = zscore(h)
        if update_object=='M':
            h_temp = h.reshape(self._x, self._y)
            self.theta += (1/tao_theta)*(h_temp*h_temp-self.theta)         
            self.M += (1/tao_M)*(np.dot(h*(h-self.theta.reshape(-1,1)),h.T))
        if update_object=='W':
            h_temp = h.reshape(self._x, self._y)
            self.theta += (1/tao_theta)*(h_temp*h_temp-self.theta) 
            for i in range(x.shape[0]):
                self._weights[:,:,i] += (1/tao_M)*(h_temp*(h_temp-self.theta)*x[i])
        
    def STDP_update(self, Output_current, tao, a, b):
        """Spike-timing model of plasticity"""
        for i in tqdm(range(self.M.shape[0])):
            for j in range(self.M.shape[1]):
                '''j->i'''
                if self.M[i,j] != 0:
                    i_pos = np.unravel_index(i, (self._x,self._y))
                    j_pos = np.unravel_index(j, (self._x,self._y))
                    i_spikes = Output_current[i_pos[0],i_pos[1],:]
                    j_spikes = Output_current[j_pos[0],j_pos[1],:]
                    delta_M = 0
                    for t_i,v_i in enumerate(i_spikes):
                        if v_i==1:
                            for t_j,v_j in enumerate(j_spikes):
                                if v_j==1:
                                    if t_i>t_j:
                                        delta_M += a * np.exp(-(t_i-t_j)/tao)
                                    if t_i<t_j:
                                        delta_M -= b * np.exp(-(t_j-t_i)/tao) 
                    self.M[i,j] += delta_M
                else:
                    pass
    
    
    #### Visulization ####
    def U_avg_matrix(self):
        heatmap = self.distance_map()
        plt.figure(figsize=(7, 7))
        plt.title('U-avg-matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        return heatmap
    
    def U_onefeature_avg_matrix(self, feature):
        def distance_map(self):
            um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
            it = np.nditer(um, flags=['multi_index'])
            while not it.finished:
                for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                    for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                        if (ii >= 0 and ii < self._weights.shape[0] and
                                jj >= 0 and jj < self._weights.shape[1]):
                            w_1 = self._weights[ii, jj, feature]
                            w_2 = self._weights[it.multi_index][feature]
                            um[it.multi_index] += fast_norm(w_1-w_2)
                it.iternext()
            um = um/um.max()
            return um
        heatmap = distance_map(self)
        plt.figure(figsize=(7, 7))
        plt.title('U_onefeature_avg_matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        return heatmap
    
    def U_min_matrix(self):
        """Returns the distance map of the weights.
        Each cell is the normalised min of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance."""
        def min_dist(self):
            um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
            it = np.nditer(um, flags=['multi_index'])
            while not it.finished:
                Dist_neig = []
                for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                    for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                        if (ii >= 0 and ii < self._weights.shape[0] and
                                jj >= 0 and jj < self._weights.shape[1]):
                            w_1 = self._weights[ii, jj, :]
                            w_2 = self._weights[it.multi_index]
                            Dist_neig.append(fast_norm(w_1-w_2))
                Dist_neig.remove(0)
                um[it.multi_index] = np.min(Dist_neig)
                it.iternext()
            um = um/um.max()
            return um
        heatmap = min_dist(self)
        plt.figure(figsize=(7, 7))
        plt.title('U-min-matrix')
        plt.imshow(heatmap, cmap=plt.get_cmap('bone_r'))
        plt.colorbar()
        return heatmap
        
    def Component_Plane(self, feature_index):
        """
        Component_Plane表示了map里每个位置的神经元对什么特征最敏感(或者理解为与该特征取值最匹配)
        """
        plt.figure(figsize=(7, 7))
        plt.title('Component Plane: feature_index is %d' % feature_index)
        plt.imshow(self._weights[:,:,feature_index], cmap='coolwarm')
        plt.colorbar()
        plt.show()
        
    def Winners_map(self, data, blur=None):
        if blur == None:
            plt.figure()
            plt.imshow(self.activation_response(data))
            plt.colorbar()
        if blur == 'GB':
            img = self.activation_response(data)
            plt.figure()
            plt.imshow(cv2.GaussianBlur(img,(5,5),0))
            plt.colorbar()
            
    def dynamics_pattern(self, data, represent_type='neuron'):
        def F(x):
            return np.where(x<=0, 0, 0.8*x)
        def diff_equation(v, t, M, H, tao, noise_level):
            h = H
            Change = F((np.dot(M,v)+h))
            dvdt = tao * (-v + Change)
            return dvdt
        times = np.linspace(0,10,100)
        SOM_act = self.activate(data)
        H = 1/SOM_act.reshape(-1)
        solution = odeint(diff_equation, H, times, args=(self.M, H, 1.2, 0))        
        
        if represent_type=='neuron':
            plt.figure()
            neurons = np.random.choice(np.arange(self.M.shape[0]), 100)
            for neuron in neurons:
                plt.plot(solution[:,neuron])
                
        if represent_type=='pattern':
            plt.figure()
            plt.ion()     # 开启一个画图的窗口
            for i in range(solution.shape[0]):
                plt.imshow(solution[i,:].reshape(64,64), 'jet')
                plt.title('This is time: %d' %i)
                plt.axis('off')
                plt.pause(0.000000000000000001)       # 停顿时间
            plt.pause(0)   # 防止运行结束时闪退
            
        if represent_type=='state_space_pca':
            pca = PCA(n_components=3)
            solution_pca = pca.fit_transform(solution[:100,:])
            
            plt.figure(figsize=(7,7))
            x = solution_pca[:,0]
            y = solution_pca[:,1]
            z = solution_pca[:,2]
            plt.ion()
            for i in range(100):
                ax = plt.axes(projection='3d')
                ax.plot3D(x[:i], y[:i], z[:i])  
                ax.set_xlim(solution_pca[:,0].min(), solution_pca[:,0].max())
                ax.set_ylim(solution_pca[:,1].min(), solution_pca[:,1].max())
                ax.set_zlim(solution_pca[:,2].min(), solution_pca[:,2].max())
                ax.grid(False)
                plt.title('This is time: %d' %i)
                plt.pause(0.01)  
            plt.pause(0) 





if __name__ == '__main__':
    data = np.genfromtxt('/Users/mac/Desktop/TDCNN/minisom/examples/iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

    # Initialization and training
    som = VTCSOM(7, 7, 4, sigma=3, learning_rate=0.5, 
                      sigma_decay_function=none_decay, lr_decay_function=asymptotic_decay,
                      neighborhood_function='gaussian')
    som.pca_weights_init(data)
    q_error, t_error = som.Train(data, 100, verbose=False)
    
    plt.figure()
    plt.plot(q_error)
    plt.ylabel('quantization error')
    plt.xlabel('iteration index')
    
    plt.figure()
    plt.plot(t_error)
    plt.ylabel('som.topographic error')
    plt.xlabel('iteration index')
    
    


                