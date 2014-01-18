# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:47:31 2013

@author: daredavil
"""

import sys
sys.path.append(r'C:\Users\daredavil\Documents\Python Scripts\RBMver2')

import cPickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from RBM import GaussRBM, BinRBM, step_iterator, batch_func_generator
from NeuralNetwork import NeuralNetwork

save_dict = {}
bas = 0
son = 10000
X_train = np.zeros((50000,3072))
y_train = np.zeros(50000)
file_name = r'cifar-10-batches-py\data_batch_1'
for i in xrange(1,6,1):
    file_name = file_name[:-1] + str(i)
    with open(file_name, 'rb') as f_:
        data_dict = cPickle.load(f_)
        data = data_dict['data']
        labels = data_dict['labels']
        X_train[bas:son] = data
        y_train[bas:son] = np.array(labels)
        bas = son
        son += 10000

file_name = r'cifar-10-batches-py\test_batch'
with open(file_name, 'rb') as f_:
    data_dict = cPickle.load(f_)
    X_test = data_dict['data']
    y_test = np.array(data_dict['labels'])
    
file_name = r'cifar-10-batches-py\batches.meta'
with open(file_name, 'rb') as f_:
    data_dict = cPickle.load(f_)
    label_names = data_dict['label_names']
    
X_all = np.vstack((X_train,X_test))
scl = StandardScaler()
X_all = scl.fit_transform(X_all)
X_train = X_all[:50000]
X_test = X_all[50000:]
del X_all
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

## 3072-8192, 8192-4096, 4096-2048, 2048-1024, 1024-512
random_state = 500
RBMlayers_dict = {0:     {
                            'dimension':X_train.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'linear',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':8192,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         } }
learning_rate, weight_decay, momentum= step_iterator(1,1,0), step_iterator(0,0,0), step_iterator(0,0,0)
batch_func = batch_func_generator(X_train, batch_size = 100)
rbm = GaussRBM(layers_dict = RBMlayers_dict, weight_list = None, random_state = random_state)
print 'Training starts'
rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 50, verbose = True)
      #          sparsity_cond = True, sparsity_target = 0.01, sparsity_lambda = 1e-6)
                
rbm_list = []
rbm_list.append(rbm)

dimen = 8192
for _ in xrange(4):
    dimen = dimen/2
    X_train = rbm.transform(X_train)
    RBMlayers_dict = {0:     {
                                'dimension':X_train.shape[1],     
                                'bias': None,
                                'value': None,
                                'layer_type': 'binary',
                                'layer_name': 'input'
                            },
                            1: {
                                'dimension':dimen,     
                                'bias': None,
                                'value': None,
                                'layer_type': 'binary',
                                'layer_name': 'hidden'
                                } }
    learning_rate, weight_decay, momentum= step_iterator(0.1,0.01,-0.02), step_iterator(0,0,0), step_iterator(0.1,0.9,0.1)
    batch_func = batch_func_generator(X_train, batch_size = 100)
    rbm = BinRBM(layers_dict = RBMlayers_dict, weight_list = None, random_state = random_state)
    print 'Training starts'
    rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                    weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 50, verbose = True)
    rbm_list.append(rbm)
    
X_train = X_train_copy
for rbm in rbm_list:
    X_train = rbm.transform(X_train)
    X_test = rbm.transform(X_test)
save_dict['rbmtrain'] = X_train
save_dict['rbmtest'] = X_test

X_test = X_test_copy
X_train = X_train_copy
Networklayer_dict = {0:
        { 'n_neuron': X_train.shape[1],
          'incoming_layer_list': [],
          'incoming_weight_list': [],
          'bias': None,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None, 'random_state':random_state,
          'layer_type': 'input',
          'back_error': 0,
          'link2input': X_train,
          'link2target': None},
    1:  { 'n_neuron': 8192,
          'incoming_layer_list': [0,],
          'incoming_weight_list': [rbm_list[0].weight_list[0].T,],
          'bias': rbm_list[0].hidden_layer.bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None,
          'layer_type': 'hidden', 'random_state':random_state,
          'back_error': 0 },
    2:  { 'n_neuron': 4096,
          'incoming_layer_list': [1,],
          'incoming_weight_list': [rbm_list[1].weight_list[0].T,],
          'bias': rbm_list[1].hidden_layer.bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None,
          'layer_type': 'hidden', 'random_state':random_state,
          'back_error': 0 },
    3:  { 'n_neuron': 2048,
          'incoming_layer_list': [2,],
          'incoming_weight_list': rbm_list[2].weight_list[0].T,
          'bias': rbm_list[2].hidden_layer.bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None,
          'layer_type': 'hidden', 'random_state':random_state,
          'back_error': 0},
    4:  { 'n_neuron': 1024,
          'incoming_layer_list': [3,],
          'incoming_weight_list': rbm_list[3].weight_list[0].T,
          'bias':  rbm_list[3].hidden_layer.bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None,
          'layer_type': 'hidden', 'random_state':random_state,
          'back_error': 0},
    5:  { 'n_neuron': 512,
          'incoming_layer_list': [4,],
          'incoming_weight_list': rbm_list[4].weight_list[0].T,
          'bias': rbm_list[4].hidden_layer.bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid',
          'value': None,
          'layer_type': 'hidden', 'random_state':random_state,
          'back_error': 0},
    6:  { 'n_neuron': 1024,
          'incoming_layer_list': [5],
          'incoming_weight_list': rbm_list[4].weight_list[0],
          'bias': rbm_list[4].input_layer_list[0].bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid', 'random_state':random_state,
          'value': None, #'drop_rate':0.3,
          'layer_type': 'hidden',
          'back_error': 0},
    7:  { 'n_neuron': 2048,
          'incoming_layer_list': [6],
          'incoming_weight_list': rbm_list[3].weight_list[0],
          'bias': rbm_list[3].input_layer_list[0].bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid', 'random_state':random_state,
          'value': None, #'drop_rate':0.3,
          'layer_type': 'hidden',
          'back_error': 0},
    8:  { 'n_neuron': 4096,
          'incoming_layer_list': [7],
          'incoming_weight_list': rbm_list[2].weight_list[0],
          'bias': rbm_list[2].input_layer_list[0].bias,
          'loss': 'cross_entropy',
          'act_func_name': 'sigmoid', 'random_state':random_state,
          'value': None, #'drop_rate':0.3,
          'layer_type': 'hidden',
          'back_error': 0},
    9: {
      'n_neuron': 8192,
      'incoming_layer_list': [8,],
      'incoming_weight_list': rbm_list[1].weight_list[0],
      'bias': rbm_list[1].input_layer_list[0].bias,
      'loss': 'cross_entropy',
      'act_func_name': 'sigmoid',
      'value': None,
      'layer_type': 'hidden', 'random_state':random_state,
      'back_error': 0,},
    10: {
      'n_neuron': X_train.shape[1],
      'incoming_layer_list': [9,],
      'incoming_weight_list': rbm_list[0].weight_list[0],
      'bias': rbm_list[0].input_layer_list[0].bias,
      'loss': 'mse',
      'act_func_name': 'linear',
      'value': None,
      'layer_type': 'output', 'random_state':random_state,
      'back_error': 0,  
      'link2target': X_train
       } }
network = NeuralNetwork(n_layers=11, layer_dict = Networklayer_dict)
network.fit(batch_size = 1000, learning_rate = step_iterator(0.1,0.01,-0.02), 
            weight_decay = step_iterator(0,0,0), momentum = step_iterator(0,0,0), n_iter = 5, switch_point = None)
network.transform([X_train,])[0]
hidd_rep_train = network.layer_list[5].value

network.transform([X_test,])[0]
hidd_rep_test = network.layer_list[5].value

save_dict['hrtrain'] = hidd_rep_train
save_dict['hrtest'] = hidd_rep_test

with open('data.pkl', 'wb') as f_:
    cPickle.dump(save_dict, f_)





