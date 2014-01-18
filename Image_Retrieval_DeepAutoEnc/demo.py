# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:33:08 2013

@author: daredavil
"""

from __future__ import division
import numpy as np
import cPickle
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button

class Draw:
    def __init__(self, N):
        self.N = N
        self.fig, self.axis = plt.subplots(ncols = N+1)
        
    def draw(self,event):
        draw_on_screen2(self.fig,self.axis,self.N)

def draw_on_screen2(fig, axis, turn = 5):
    sec = np.random.randint(0,9999)
    test_im = X_test[sec]
    test_org = X_test_org[sec]
    test_label = y_test[sec]
    test_org_ = np.zeros((32,32,3))
    
    inds = smp.cosine_similarity(test_im, X_train).flatten().argsort()[-turn:][::-1]
    draw_images = X_train_org[inds]
    draw_labels = label_names[y_train[inds].astype('int')]
    #fig, axs = plt.subplots(ncols = turn+1)
    test_org_[:,:,0] = test_org[:1024].reshape((32,32))
    test_org_[:,:,1] = test_org[1024:2048].reshape((32,32))
    test_org_[:,:,2] = test_org[2048:].reshape((32,32))
    axis[0].imshow(test_org_.astype('uint8'))
    axis[0].set_title('Test image\n'+label_names[test_label])
    axis[0].set_axis_off()
    for i in xrange(1,turn+1):
        draw_im_ = np.zeros((32,32,3))
        draw_im_[:,:,0] = draw_images[i-1][:1024].reshape((32,32))
        draw_im_[:,:,1] = draw_images[i-1][1024:2048].reshape((32,32))
        draw_im_[:,:,2] = draw_images[i-1][2048:].reshape((32,32))
        axis[i].imshow(draw_im_.astype('uint8'))
        axis[i].set_axis_off()
        tt = 'Turn %d\n'%i
        axis[i].set_title(tt + draw_labels[i-1])
    plt.draw()

def draw_on_screen(X_train, X_test, y_train, y_test, X_train_org, X_test_org,label_names, turn = 5):
    sec = np.random.randint(0,9999)
    test_im = X_test[sec]
    test_org = X_test_org[sec]
    test_label = y_test[sec]
    test_org_ = np.zeros((32,32,3))
    
    inds = smp.cosine_similarity(test_im, X_train).flatten().argsort()[-turn:][::-1]
    draw_images = X_train_org[inds]
    draw_labels = label_names[y_train[inds].astype('int')]
    fig, axs = plt.subplots(ncols = turn+1)
    test_org_[:,:,0] = test_org[:1024].reshape((32,32))
    test_org_[:,:,1] = test_org[1024:2048].reshape((32,32))
    test_org_[:,:,2] = test_org[2048:].reshape((32,32))
    axs[0].imshow(test_org_.astype('uint8'))
    axs[0].set_title('Test image\n'+label_names[test_label])
    for i in xrange(1,turn+1):
        draw_im_ = np.zeros((32,32,3))
        draw_im_[:,:,0] = draw_images[i-1][:1024].reshape((32,32))
        draw_im_[:,:,1] = draw_images[i-1][1024:2048].reshape((32,32))
        draw_im_[:,:,2] = draw_images[i-1][2048:].reshape((32,32))
        axs[i].imshow(draw_im_.astype('uint8'))
        tt = 'Turn %d\n'%i
        axs[i].set_title(tt + draw_labels[i-1])
    plt.axis('off')
    plt.show()
    

bas = 0
son = 10000
X_train_org = np.zeros((50000,3072))
y_train = np.zeros(50000)
file_name = r'cifar-10-batches-py\data_batch_1'
for i in xrange(1,6,1):
    file_name = file_name[:-1] + str(i)
    with open(file_name, 'rb') as f_:
        data_dict = cPickle.load(f_)
        data = data_dict['data']
        labels = data_dict['labels']
        X_train_org[bas:son] = data
        y_train[bas:son] = np.array(labels)
        bas = son
        son += 10000
        
#im = np.zeros((32,32,3))
#fig, axis = plt.subplots(figsize = (2,2))
#im[:,:,0] = X_train_org[0][:1024].reshape((32,32))
#im[:,:,1] = X_train_org[0][1024:2048].reshape((32,32))
#im[:,:,2] = X_train_org[0][2048:].reshape((32,32))
#axis.imshow(im[0].reshape((32,32,3)).astype('uint8'))
#plt.show()
#raise

file_name = r'cifar-10-batches-py\test_batch'
with open(file_name, 'rb') as f_:
    data_dict = cPickle.load(f_)
    y_test = np.array(data_dict['labels'])
    X_test_org = data_dict['data']


file_name = r'D:\contentfinaldata\data.pkl'
with open(file_name, 'rb') as f_:
    data_dict = cPickle.load(f_)
    
X_train = data_dict['rbmtrain']
X_test = data_dict['rbmtest']

file_name = r'cifar-10-batches-py\batches.meta'
with open(file_name, 'rb') as f_:
    data_dict = cPickle.load(f_)
    label_names = np.array(data_dict['label_names'])

dd = Draw(N = 5)
ax = plt.axes([0.7, 0.05, 0.1, 0.075])
bx = Button(ax,'draw')
bx.on_clicked(dd.draw)
plt.show()
#draw_on_screen(X_train, X_test, y_train, y_test, X_train_org, X_test_org,label_names, turn = 5)