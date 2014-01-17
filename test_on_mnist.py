# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 16:09:47 2013

@author: daredavil
"""

from __future__ import division, print_function
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from RBM import BinRBM, step_iterator, batch_func_generator
from NeuralNetwork import NeuralNetwork
import pickle

# load mnist data
mnist = fetch_mldata('MNIST original')
X = mnist.data / 255.0
y = mnist.target

#X = (X - X.mean(axis = 0))/X.std(axis = 0)
#X[np.isnan(X)] = 0
#X[np.isinf(X)] = 0

np.random.seed(4)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]
X_train, X_test, y_train, y_test = X[:-10000], X[-10000:], y[:-10000], y[-10000:]
y_train = LabelBinarizer().fit_transform(y_train)
y_test =  LabelBinarizer().fit_transform(y_test)
X_train_copy = X_train.copy()
#---------------------------------------------------------

#dbn initialization
RBM1layers_dict = {0: {
                            'dimension':X.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':500,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
RBM2layers_dict = {0: {
                            'dimension':RBM1layers_dict[1]['dimension'],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':500,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
RBM3layers_dict = {0: {
                            'dimension':RBM2layers_dict[1]['dimension'],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':2000,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
learning_rate, weight_decay, momentum= step_iterator(0.1,0.1,0), step_iterator(2e-4,2e-4,0), step_iterator(0.5,0.9,0.05)
batch_func = batch_func_generator(X_train, batch_size = 100)
rbm0 = BinRBM(layers_dict = RBM1layers_dict, weight_list = None, random_state = None)
rbm0.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 50, verbose = True)
X_train = rbm0.transform(X_train, sample = False)

learning_rate, weight_decay, momentum= step_iterator(0.1,0.1,0), step_iterator(2e-4,2e-4,0), step_iterator(0.5,0.9,0.05)
batch_func = batch_func_generator(X_train, batch_size = 100)
rbm1 = BinRBM(layers_dict = RBM2layers_dict, weight_list = None, random_state = None)
rbm1.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 50, verbose = True)
X_train = rbm1.transform(X_train, sample = False)

learning_rate, weight_decay, momentum= step_iterator(0.1,0.1,0), step_iterator(2e-4,2e-4,0), step_iterator(0.5,0.9,0.05)
batch_func = batch_func_generator(X_train, batch_size = 100)
rbm2 = BinRBM(layers_dict = RBM3layers_dict, weight_list = None, random_state = None)
rbm2.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 50, verbose = True)
                
Networklayer_dict = {0:
    { 'n_neuron': rbm2.hidden_layer.dimension,
      'incoming_layer_list': [],
      'incoming_weight_list': [],
      'bias': None,
      'loss': 'cross_entropy',
      'act_func_name': 'sigmoid',
      'value': None,
      'layer_type': 'input',
      'back_error': 0,
      'link2input': rbm2.transform(X_train, sample = False),
      'link2target': None
    }, 1: {
      'n_neuron': y_train.shape[1],
      'incoming_layer_list': [0,],
      'incoming_weight_list': [],
      'bias': None,
      'loss': 'cross_entropy',
      'act_func_name': 'softmax',
      'value': None,
      'layer_type': 'output',
      'back_error': 0,
      'link2input': None,
      'link2target': y_train } }

network = NeuralNetwork(n_layers=2, layer_dict = Networklayer_dict)
network.fit(batch_size = 1000, learning_rate = step_iterator(0.1,0.01,-0.02), 
            weight_decay = step_iterator(0,0,0), momentum = step_iterator(0.1,0.9,0.1), n_iter = 100, switch_point = 10)

y_pred = network.transform(rbm2.transform(rbm1.transform(rbm0.transform(X_test))))[0]
correct = np.sum(y_pred.argmax(axis=1) == y_test.argmax(axis=1))
print('correct = %d in %d'%(correct,X_test.shape[0]))
network.transform(rbm2.transform(rbm1.transform(rbm0.transform(X_train_copy))))[0]
error = network.empirical_error(target = y_train)
print('initial error: %f'%error)

with open(r"C:\Users\daredavil\Documents\Python Scripts\RBMver2\rbms.pkl",'wb') as file_:
    pickle.dump((rbm0.hidden_layer.dimension, rbm0.weight_list[0], rbm0.hidden_layer.bias,
                 rbm1.hidden_layer.dimension, rbm1.weight_list[0], rbm1.hidden_layer.bias,
                 rbm2.hidden_layer.dimension, rbm2.weight_list[0], rbm2.hidden_layer.bias,
                 network.output_layer_list[0].incoming_weight_list[0], network.output_layer_list[0].bias), file_)


Networklayer_dict = { 0: { 
                                'n_neuron': X.shape[1],    
                                'incoming_layer_list': [],
                                'incoming_weight_list': [],
                                'bias': None,
                                'loss': 'cross_entropy',
                                'act_func_name': 'sigmoid',
                                'value': None,
                                'layer_type': 'input',
                                'back_error' : 0,
                                'link2input': X_train_copy,
                                'link2target': None,
                                'random_state': None
                              },
                      1: { 
                                'n_neuron': rbm0.hidden_layer.dimension,    
                                'incoming_layer_list': [0,],
                                'incoming_weight_list': rbm0.weight_list[0].T,
                                'bias': rbm0.hidden_layer.bias,
                                'loss': 'cross_entropy',
                                'act_func_name': 'sigmoid',
                                'value': None,
                                'layer_type': 'hidden',
                                'back_error' : 0,
                                'link2input': None,
                                'link2target': None,
                                'random_state': None
                              },
                      2: { 
                                'n_neuron': rbm1.hidden_layer.dimension,    
                                'incoming_layer_list': [1,],
                                'incoming_weight_list': rbm1.weight_list[0].T,
                                'bias': rbm1.hidden_layer.bias,
                                'loss': 'cross_entropy',
                                'act_func_name': 'sigmoid',
                                'value': None,
                                'layer_type': 'hidden',
                                'back_error' : 0,
                                'link2input': None,
                                'link2target': None,
                                'random_state': None
                              },
                      3: { 
                                'n_neuron': rbm2.hidden_layer.dimension,    
                                'incoming_layer_list': [2,],
                                'incoming_weight_list': rbm2.weight_list[0].T,
                                'bias': rbm2.hidden_layer.bias,
                                'loss': 'cross_entropy',
                                'act_func_name': 'sigmoid',
                                'value': None,
                                'layer_type': 'hidden',
                                'back_error' : 0, 'drop_rate':0.5,
                                'link2input': None,
                                'link2target': None,
                                'random_state': None
                              },
                      4: { 
                                'n_neuron': y_train.shape[1],    
                                'incoming_layer_list': [3,],
                                'incoming_weight_list': [network.output_layer_list[0].incoming_weight_list[0],],
                                'bias': network.output_layer_list[0].bias,
                                'loss': 'cross_entropy',
                                'act_func_name': 'softmax',
                                'value': None,
                                'layer_type': 'output',
                                'back_error' : 0,
                                'link2input': None,
                                'link2target': y_train,
                                'random_state': None
                              }
                  }
                  
network = NeuralNetwork(n_layers=5, layer_dict = Networklayer_dict)
network.fit(batch_size = 1000, learning_rate = step_iterator(1e-4,1e-6,-1e-5), 
            weight_decay = step_iterator(0,0,0), momentum = step_iterator(0,0,0), n_iter = 100, switch_point = None)
#for layer in network.input_layer_list:
#    layer.value = X_test
#y_pred = network.feed_forward()[0]
y_pred = network.transform(X_test)[0]
correct = np.sum(y_pred.argmax(axis=1) == y_test.argmax(axis=1))
print('correct = %d in %d'%(correct,X_test.shape[0]))