# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:18:59 2013

@author: Ümit Ekmekçi, Esra Gülþen
"""

import h5py
import numpy as np
import scipy.io as sio
import pickle
from RBM import GaussRBM, BinRBM, step_iterator, batch_func_generator
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn import svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
#from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA



with h5py.File(r'C:\Users\daredavil\Documents\MATLAB\Poje2MCA\readyfordeep_raw.mat','r') as f:
    labels_train = np.array(f[u'labels_train']).flatten()
    labels_test = np.array(f[u'labels_test']).flatten()
    G_train = np.array(f[u'G_train']).T
    G_test = np.array(f[u'G_test']).T
    B_train = np.array(f[u'B_train']).T
    B_test = np.array(f[u'B_test']).T
    R_test = np.array(f[u'R_test']).T
    R_train = np.array(f[u'R_train']).T

labels_test -= 1
labels_train -= 1


scalerR = preprocessing.StandardScaler()
scalerG = preprocessing.StandardScaler()
scalerB = preprocessing.StandardScaler()
n_R,n_G,n_B = R_train.shape[0], G_train.shape[0], B_train.shape[0]    
R = scalerR.fit_transform(np.vstack((R_train,R_test)))
R_train, R_test = R[:n_R], R[n_R:]
G = scalerG.fit_transform(np.vstack((G_train,G_test)))
G_train, G_test = G[:n_G], G[n_G:]
B = scalerB.fit_transform(np.vstack((B_train,B_test)))
B_train, B_test = B[:n_B], B[n_B:]

clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2, weights='uniform')
clf.fit(np.hstack((R_train,G_train,B_train)), labels_train)
y_pred = clf.predict(np.hstack((R_test,G_test,B_test)))
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#674 dogru

random_state = 500

RBMRlayers_dict = {0:     {
                            'dimension':R_train.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'linear',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':500,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         } }
RBMGlayers_dict = {0:     {
                            'dimension':G_train.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'linear',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':500,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         } }
RBMBlayers_dict = {0:     {
                            'dimension':B_train.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'linear',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':500,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         } }

learning_rate, weight_decay, momentum= step_iterator(1,1,0), step_iterator(0,0,0), step_iterator(0,0,0)
batch_func = batch_func_generator(R_train, batch_size = 50)
rbmR = GaussRBM(layers_dict = RBMRlayers_dict, weight_list = None, random_state = random_state)
print 'Training starts'
rbmR.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 20, verbose = True)
      #          sparsity_cond = True, sparsity_target = 0.01, sparsity_lambda = 1e-6)
batch_func = batch_func_generator(G_train, batch_size = 50)
rbmG = GaussRBM(layers_dict = RBMGlayers_dict, weight_list = None, random_state = random_state)
print 'Training starts'
rbmG.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 20, verbose = True)
     #           sparsity_cond = True, sparsity_target = 0.01, sparsity_lambda = 1e-6)
batch_func = batch_func_generator(B_train, batch_size = 50)
rbmB = GaussRBM(layers_dict = RBMBlayers_dict, weight_list = None, random_state = random_state)
print 'Training starts'
rbmB.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 20, verbose = True)
         #       sparsity_cond = True, sparsity_target = 0.01, sparsity_lambda = 1e-6)

rbmR_hidden, rbmG_hidden, rbmB_hidden = rbmR.transform(R_train), rbmG.transform(G_train), rbmB.transform(B_train)
rbmR_hidden_test, rbmG_hidden_test, rbmB_hidden_test = rbmR.transform(R_test), rbmG.transform(G_test), rbmB.transform(B_test)

clf = svm.SVC(kernel = 'linear', C = 100, random_state=random_state)
clf.fit(np.hstack((rbmR_hidden, rbmG_hidden, rbmB_hidden)),labels_train)
y_pred = clf.predict(np.hstack((rbmR_hidden_test, rbmG_hidden_test, rbmB_hidden_test)))
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()

clf = svm.SVC(kernel = 'rbf', gamma = 0.01, C = 10, random_state=random_state)
clf.fit(np.hstack((rbmR_hidden, rbmG_hidden, rbmB_hidden)),labels_train)
y_pred = clf.predict(np.hstack((rbmR_hidden_test, rbmG_hidden_test, rbmB_hidden_test)))
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()



RBMlayers_dict = {0:     {
                            'dimension':rbmR.hidden_layer.dimension,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':rbmG.hidden_layer.dimension,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      2: {
                            'dimension':rbmB.hidden_layer.dimension,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      3: {
                            'dimension':1000,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
learning_rate, weight_decay, momentum= step_iterator(0.1,0.01,-0.002), step_iterator(1e-6,1e-6,0), step_iterator(0.1,0.9,0.05)
batch_func = batch_func_generator([rbmR_hidden, rbmG_hidden, rbmB_hidden], batch_size = 50)
rbm = BinRBM(layers_dict = RBMlayers_dict, weight_list = None, random_state = random_state)
print 'Training starts'
rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 20, verbose = True)
        #        sparsity_cond = True, sparsity_target = 0.01, sparsity_lambda = 1e-6)
                
###sample from model####
#rbm.run_gibbs(X = [rbmR_hidden[1000],rbmG_hidden[1000],rbmB_hidden[1000]], step_size = 1, sample = False)
#rbm.run_gibbs(X = [np.zeros(500),np.zeros(500),np.zeros(500)], step_size = 1000, sample = False)
#hidR,hidG,hidB = rbm.input_layer_list[0].value,rbm.input_layer_list[1].value,rbm.input_layer_list[2].value
#rbmR.hidden_layer.value,rbmG.hidden_layer.value,rbmB.hidden_layer.value = hidR,hidG,hidB
#rbmR.feed_back(sample = True)
#rbmG.feed_back(sample = True)
#rbmB.feed_back(sample = True)
rbmR.run_gibbs(np.random.rand(1800)*10 -5, step_size = 1000, sample = True)
rbmG.run_gibbs(np.random.rand(1800)*10 -5, step_size = 1000, sample = True)
rbmB.run_gibbs(np.random.rand(1800)*10 -5, step_size = 1000, sample = True)
visR = scalerR.inverse_transform(rbmR.input_layer_list[0].value)
visG = scalerG.inverse_transform(rbmG.input_layer_list[0].value)
visB = scalerB.inverse_transform(rbmB.input_layer_list[0].value)
visRR = scalerR.inverse_transform(R_train[1000])
visGG = scalerG.inverse_transform(G_train[1000])
visBB = scalerB.inverse_transform(B_train[1000])
#sio.savemat(r'C:\Users\daredavil\Documents\MATLAB\Poje2MCA\sample_new4.mat',{'R':visR, 'G':visG,'B':visB, 'RR':visRR, 'GG':visGG,'BB':visBB,
#                                                                            'masks':rbmR.weight_list[0][[30,70,80,150,220,380,420],:]})
#########################                

combined = rbm.transform([rbmR_hidden, rbmG_hidden, rbmB_hidden])
combined_test = rbm.transform([rbmR_hidden_test, rbmG_hidden_test, rbmB_hidden_test])


sonuclar = {'sgd':0,'svm_lin':0,'svm_rbf':0,'knn':0,'nn':[],'svm_lin_n':0,'svm_rbf_n':0}
parameters = {'alpha':[1e-1,1e-2,1e-3,1e-4,1e-5]}
clf = GridSearchCV(SGDClassifier(loss="log", penalty="l2", n_iter = 100,random_state=random_state), parameters) 
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_
sonuclar['sgd'] = (y_pred == labels_test).sum()

#parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#clf = GridSearchCV(svm.SVC(C=1,random_state=random_state,), parameters)
class_weight = dict(zip([0,1,2,3,4,5,6],[100,100,50,10,10,10]))
clf = svm.SVC(kernel = 'linear',random_state=random_state,C=10, class_weight = class_weight)
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.n_support_
sonuclar['svm_lin'] = (y_pred == labels_test).sum()
sonuclar['svm_lin_n'] = clf.n_support_

#parameters = {'C':[1,10,100], 'gamma':[0.1,0.01,0.001]}
#clf = GridSearchCV(svm.SVC(kernel = 'rbf',random_state=random_state, class_weight = class_weight), parameters)
clf = svm.SVC(kernel = 'rbf', gamma = 0.01, random_state=random_state,C = 10)
clf.fit(combined,labels_train)
svm_pred_test = clf.predict(combined_test)
svm_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (svm_pred_test == labels_test).sum()
#print clf.best_estimator_
print clf.n_support_
print confusion_matrix(labels_test, svm_pred_test)
sonuclar['svm_rbf'] = (svm_pred_test == labels_test).sum()
sonuclar['svm_rbf_n'] = clf.n_support_
#sio.savemat(r'C:\Users\daredavil\Documents\MATLAB\Poje2MCA\svm_sonuc.mat',{'svm_train':svm_pred_train,'svm_test':svm_pred_test})

#parameters = {'n_neighbors':[1,3,5,7], 'weights':['uniform', 'distance']}
#parameters =  {'leaf_size':[10,20]}
#clf = GridSearchCV(neighbors.KNeighborsClassifier(n_neighbors = 1, weights = 'uniform'), parameters)
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, weights = 'uniform', leaf_size = 100)
clf.fit(combined, labels_train)
knn_pred_test = clf.predict(combined_test)
knn_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (knn_pred_test == labels_test).sum()
#print clf.best_estimator_
#sio.savemat(r'C:\Users\daredavil\Documents\MATLAB\Poje2MCA\predictions.mat',{'svm_train':svm_pred_train+1, 'knn_train':knn_pred_train+1,
#                                                                             'svm_test':svm_pred_test+1, 'knn_test':knn_pred_test+1})
print confusion_matrix(labels_test, knn_pred_test)
sonuclar['knn'] = (knn_pred_test == labels_test).sum()



clf = tree.DecisionTreeClassifier(random_state=random_state)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()

#parameters = {'n_estimators':[10,20,30]}
#clf = GridSearchCV(RandomForestClassifier(n_estimators = 20,random_state=random_state), parameters)
clf = RandomForestClassifier(n_estimators = 40,random_state=random_state)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_

parameters = {'C':[1,1e1]}
clf = GridSearchCV(LogisticRegression(random_state=random_state),parameters)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_
w_list = clf.best_estimator_.coef_.T
w_bias = clf.best_estimator_.intercept_