# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 01:09:28 2013

@author: Ümit Ekmekçi, Esra Gülþen
"""

import h5py
import numpy as np
#import scipy.io as sio
import pickle
from RBM import BinRBM, step_iterator, batch_func_generator
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn import svm, neighbors, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


def binTransforemer(k, seq):
    y = np.zeros((seq.shape[0],k))
    y[np.arange(seq.shape[0]),seq.astype('int')] = 1
    return y


with h5py.File(r'C:\Users\daredavil\Documents\MATLAB\Poje2MCA\readyfordeep.mat','r') as f:
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


X_train = np.hstack((R_train,G_train,B_train))
X_test = np.hstack((R_test,G_test,B_test))
n_X = X_train.shape[0]
X = preprocessing.scale(np.vstack((X_train, X_test)))
X_train, X_test = X[:n_X], X[n_X:]
del X

combined = X_train
combined_test = X_test



#parameters = {'alpha':[1e-1,1e-2,1e-3,1e-4,1e-5]}
#clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="l2", n_iter = 10,random_state = 10), parameters) 
clf = SGDClassifier(loss="hinge", penalty="l2", n_iter = 10,random_state = 500,alpha = 1e-5)
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_
X_train_bin =  binTransforemer(labels_train.max() + 1, y_pred_train)
X_test_bin = binTransforemer(labels_test.max() + 1, y_pred)

#parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#clf = GridSearchCV(svm.SVC(C=1,random_state = 10), parameters)
clf = svm.SVC(C=10,kernel = 'rbf',gamma = 0.001, random_state=500)
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_
X_train_bin = np.hstack((X_train_bin, binTransforemer(labels_train.max() + 1, y_pred_train)))
X_test_bin = np.hstack((X_test_bin, binTransforemer(labels_test.max() + 1, y_pred)))

"""
clf = svm.SVC()
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
"""

#parameters = {'n_neighbors':[1,3,5,7], 'weights':['uniform', 'distance']}
#clf = GridSearchCV(neighbors.KNeighborsClassifier(random_state = 10), parameters)
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2, weights='uniform')
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_
X_train_bin = np.hstack((X_train_bin, binTransforemer(labels_train.max() + 1, y_pred_train)))
X_test_bin = np.hstack((X_test_bin, binTransforemer(labels_test.max() + 1, y_pred)))

clf = tree.DecisionTreeClassifier(random_state = 500)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
X_train_bin = np.hstack((X_train_bin, binTransforemer(labels_train.max() + 1, y_pred_train)))
X_test_bin = np.hstack((X_test_bin, binTransforemer(labels_test.max() + 1, y_pred)))

#parameters = {'n_estimators':[10,20,30]}
#clf = GridSearchCV(RandomForestClassifier(n_estimators = 20,random_state = 10), parameters)
clf = RandomForestClassifier(n_estimators = 30,random_state = 500)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_
X_train_bin = np.hstack((X_train_bin, binTransforemer(labels_train.max() + 1, y_pred_train)))
X_test_bin = np.hstack((X_test_bin, binTransforemer(labels_test.max() + 1, y_pred)))

#parameters = {'C':[1,1e1,1e2,1e3]}
#clf = GridSearchCV(LogisticRegression(random_state = 10),parameters)
clf = LogisticRegression(C = 1,random_state = 500)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
y_pred_train = clf.predict(combined)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_
X_train_bin = np.hstack((X_train_bin, binTransforemer(labels_train.max() + 1, y_pred_train)))
X_test_bin = np.hstack((X_test_bin, binTransforemer(labels_test.max() + 1, y_pred)))


"""
Fusion with random forest
"""
#parameters = {'n_estimators':[10,20,30]}
#clf = GridSearchCV(RandomForestClassifier(n_estimators = 20,random_state = 10), parameters)
clf = RandomForestClassifier(n_estimators = 10,random_state = 500)
clf.fit(X_train_bin, labels_train)
y_pred = clf.predict(X_test_bin)
print 'Fusion1: toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
#print clf.best_estimator_

clf = AdaBoostClassifier(n_estimators=100, random_state = 500)
clf.fit(X_train_bin, labels_train)
y_pred = clf.predict(X_test_bin)
print 'Fusion1: toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()


RBMlayers_dict = {0:     {
                            'dimension':X_train_bin.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':200,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
learning_rate, weight_decay, momentum= step_iterator(0.01,0.001,-0.002), step_iterator(2e-5,2e-5,0), step_iterator(0.5,0.9,0.05)
batch_func = batch_func_generator(X_train_bin, batch_size = 100)
rbm = BinRBM(layers_dict = RBMlayers_dict, weight_list = None, random_state = None)
print 'Training starts'
rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 500, verbose = True)
combined = rbm.transform(X_train_bin)
combined_test = rbm.transform(X_test_bin)

parameters = {'alpha':[1e-1,1e-2,1e-3,1e-4,1e-5], 'n_iter': [10, 50 ,100]}
clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="l2", n_iter = 100, random_state = 500), parameters) 
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_

parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(svm.SVC(random_state = 500), parameters)
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_


clf = svm.SVC(random_state = 500, kernel = 'rbf', C = 10, gamma = 0.001)
clf.fit(combined,labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()


parameters = {'n_neighbors':[1,3,5,7], 'weights':['uniform', 'distance']}
clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_

clf = tree.DecisionTreeClassifier(random_state = 500)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()

parameters = {'n_estimators':[10,20,30]}
clf = GridSearchCV(RandomForestClassifier(n_estimators = 20,random_state = 500), parameters)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_

parameters = {'C':[1,1e1,1e2,1e3]}
clf = GridSearchCV(LogisticRegression(random_state = 500),parameters)
clf.fit(combined, labels_train)
y_pred = clf.predict(combined_test)
print 'toplam: ', labels_test.shape[0], 'dogru: ', (y_pred == labels_test).sum()
print clf.best_estimator_