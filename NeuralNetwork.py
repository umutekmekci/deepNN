# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 19:03:51 2013

@author: daredavil
"""

from __future__ import print_function
from __future__ import division
import numpy as np
from time import time
from sklearn.utils import check_random_state
from sklearn.utils.extmath import logistic_sigmoid
from sklearn.metrics import mean_squared_error
from RBM import step_iterator
import minimize

def relu_act(x):
    return np.choose(x>0, choices=[0,x])

def linear_act(x):
    return x
def softmax_act(x):
    if x.ndim == 1:
        return np.exp(x)/np.exp(x).sum()
    return np.exp(x)/np.exp(x).sum(axis = 1)[:,np.newaxis]
    
def weight_extend(obj):
    big_weight_list, big_bias_list = [], []
    for layer in obj.layer_list:
        if layer.layer_type == 'input': continue
        bw = np.hstack([weight.flatten() for weight in layer.incoming_weight_list])
        big_bias_list.append(layer.bias)
        big_weight_list.append(bw)
    big_weight = np.hstack((np.hstack(big_weight_list), np.hstack(big_bias_list)))
    return big_weight

    
def weight_compress(big_weight, obj):
    bas,son = 0,0
    for layer in obj.layer_list:
        if layer.layer_type == 'input': continue
        for i, weight in enumerate(layer.incoming_weight_list):
            size_ = weight.size
            son = bas + size_
            shape_ = weight.shape
            layer.incoming_weight_list[i] = big_weight[bas:son].reshape(shape_)
            bas = son
    for layer in obj.layer_list:
        if layer.layer_type == 'input': continue
        size_ = layer.bias.size
        son = bas + size_
        layer.bias = big_weight[bas:son]
        bas = son
            
def helper_func_eval(big_weight, obj, isnorm):
    weight_compress(big_weight, obj)
    obj.feed_and_back(isnorm)
    error = obj.empirical_error()
    big_weight_dir_list = []
    big_bias_dir_list  =[]
    for layer in obj.layer_list:
        if layer.layer_type == 'input': continue
        bd = np.hstack([direction.flatten() for direction in layer.weight_dir_list])
        big_weight_dir_list.append(bd)
        big_bias_dir_list.append(layer.bias_dir)
    big_dir = np.hstack((np.hstack(big_weight_dir_list), np.hstack(big_bias_dir_list)))
    return (error, big_dir)
    
    
def miniNetwork(x, t, w_list, b_list, act_list, act_name_list, EC, isnorm = True):
    """
    feed forward and back feed for a network defined by parameters
    returns derivatives
    """
    y = [x,]
    dy = []
    for i, (w, b, act_func) in enumerate(zip(w_list, b_list,act_list)):
        y.append(act_func(np.dot(y[i],w) + b))
        if i+1 == len(w_list): break
        if act_name_list[i] == 'linear':
            dy.append(np.ones_like(y[-1]))
        elif act_name_list[i] == 'sigmoid':
            dy.append(y[-1]*(1-y[-1]))
        elif act_name_list[i] == 'relu':
            d_temp = np.zeros_like(y[-1])
            d_temp[y[-1] > 0] = 1
            dy.append(d_temp)
    dy.append(np.ones_like(y[-1]))
    error = mini_network_error(y[-1].copy(), t.copy(),EC,act_name_list[-1])
    back_error = y[-1] - t
    if EC == 'mse' and act_name_list[-1] == 'sigmoid':
        back_error = back_error*(y[-1]*(1-y[-1]))
    dw,db = [],[]
    for y_, dy_, w in zip(y[:-1][::-1],dy[::-1], w_list[::-1]):
        p = back_error*dy_
        dw.append(np.outer(y_,p))
        db.append(p)
        back_error = np.dot(p, w.T)
    if isnorm:
        for i, dw_ in enumerate(dw):
            dw_temp = dw_/np.linalg.norm(dw_)
            dw[i] = dw_temp
            db[i] /= np.linalg.norm(db[i])
    return (dw[::-1],db[::-1],error)
    
def mini_network_error(pred,t,loss, act_func_name):
    if loss == 'mse':
        return mean_squared_error(pred,t)
    else:
        if act_func_name == 'sigmoid':
            temp_out = 1 - pred
            error = -np.log2(pred[t.astype('bool')]).sum() - np.log2(temp_out[not t.astype('bool')]).sum()
        elif act_func_name == 'softmax':
            error = -np.log2(pred[t.astype('bool')]).sum()
        else: error = None
    return error

    
class Layer:
    def __init__(self, n_neuron = 10, incoming_layer_list = [], incoming_weight_list = [], bias = None, loss = 'cross_entropy', random_state=None,
                 act_func_name='sigmoid', value = None, layer_type = 'hidden', back_error = 0, drop_rate = 0.0, link2input = None, link2target = None):
        incoming_layer_list = incoming_layer_list if type(incoming_layer_list)==type([]) else [incoming_layer_list,]
        incoming_weight_list = incoming_weight_list if type(incoming_weight_list)==type([]) else [incoming_weight_list,]
        self.n_neuron = n_neuron
        self.incoming_layer_list = incoming_layer_list if incoming_layer_list else []
        self.incoming_weight_list = incoming_weight_list if incoming_weight_list else []
        self.bias = bias
        self.loss = loss
        self.act_func_name = act_func_name
        self.value = value
        self.layer_type = layer_type
        self.back_error = back_error
        self.drop_rate = drop_rate
        self.drop_ind = None
        self.link2input = link2input
        self.link2target = link2target
        self.target = 0
        self.rng = check_random_state(random_state)
        if incoming_layer_list:
            if not incoming_weight_list:
                self.initialize_layer()
            self.weight_grad_list = list(np.zeros(len(self.incoming_weight_list)))
            self.bias_grad = 0.0
            self.bias_dir = None
            self.weight_dir_list = None
        if act_func_name == 'sigmoid':
            self.act_func = logistic_sigmoid
        elif act_func_name == 'linear':
            self.act_func = linear_act
        elif act_func_name == 'softmax':
            self.act_func = softmax_act
        elif act_func_name == 'relu':
            self.act_func = relu_act
            
    def assign_random_state(self, random_state):
        self.rng = check_random_state(random_state)
            
    def initialize_layer(self):
        for child_layer in self.incoming_layer_list:
            self.incoming_weight_list.append(0.01*self.rng.randn(child_layer.n_neuron, self.n_neuron))
        self.bias = self.rng.randn(self.n_neuron)*0.01
        
    def layer_error(self):
        if self.loss == 'mse':
            return mean_squared_error(self.value,self.target)
        else:
            if self.act_func_name == 'sigmoid':
                temp_out = 1 - self.value
                error = -np.log2(self.value[self.target.astype('bool')]).sum() - np.log2(temp_out[~self.target.astype('bool')]).sum()
            elif self.act_func_name == 'softmax':
                error = -np.log2(self.value[self.target.astype('bool')]).sum()
            else: error = None
        return error
        
    def feed_forward(self, state = 'train'):
        activation = 0
        for child_layer, weight in zip(self.incoming_layer_list, self.incoming_weight_list):
            activation += np.dot(child_layer.value, weight)
        activation += self.bias
        #activation = activation*(1-self.drop_rate) if state == 'test' else activation
        self.value = self.act_func(activation)*(1-self.drop_rate) if state == 'test' else self.act_func(activation)
        if self.drop_rate and state == 'train':
            self.drop_ind = self.rng.rand(*self.value.shape) < self.drop_rate
            self.value[self.drop_ind] = 0.0
    
    def back_prop_grads(self, dy = None, isnorm = True):
        if dy is None:
            if self.act_func_name == 'linear':
                dy = np.ones_like(self.value)
            elif self.act_func_name == 'sigmoid':
                dy = self.value*(1-self.value)
            elif self.act_func_name == 'relu':
                dy = np.zeros_like(self.value)
                dy[self.value>0] = 1.0
                
        
        p_ = self.back_error*dy
        self.back_error = 0.0
        normalizer = 1 if isnorm else p_.shape[0]
        #normalizer = 1
        if self.drop_rate:
            p_[self.drop_ind]= 0
        #    normalizer -= self.drop_ind.sum(axis = 0)
        bias_grad = p_.sum(axis = 0)/normalizer
        weight_grad_list = []
        back_error_list = []
        for child_layer, weight in zip(self.incoming_layer_list, self.incoming_weight_list):
            weight_grad_list.append(np.dot(child_layer.value.T, p_)/normalizer)
            back_error_list.append(np.dot(p_, weight.T))
        if isnorm:
            bias_grad /= np.linalg.norm(bias_grad)
            for i, weight in enumerate(weight_grad_list):
                weight_grad_list[i] /= np.linalg.norm(weight)
        return (weight_grad_list, bias_grad, back_error_list)
        
    def update_weight_bias(self, weight_grad_list, bias_grad, lr = 0.1, weight_decay=0, momentum = 0):
        for i, weight in enumerate(self.incoming_weight_list):
            self.weight_grad_list[i] = lr*(1-momentum)*weight_grad_list[i] - momentum*self.weight_grad_list[i] + weight_decay*weight
            self.incoming_weight_list[i] -= self.weight_grad_list[i]
        self.bias_grad = lr*(1-momentum)*bias_grad - momentum*self.bias_grad
        self.bias -= self.bias_grad
        
    
class NeuralNetwork:
    
    def __init__(self, n_layers, layer_dict):
        """
        n_layers: max layer number, ex: 5
        layer_dict: see example
        layer_dict{ layer_no: { 
                                n_neuron: ?        
                                incoming_layer_list: [..]
                                incoming_weight_list: [..]
                                bias: ?
                                loss = 'cross_entropy'
                                act_func_name: ? sigmoid, linear, softmax
                                value: ?
                                layer_type: ?
                                back_error : 0
                                drop_rate: 0.0
                                link2input: None
                                link2target: None
                              }
                  }
        layer_no should start from 0
        """        
        self.n_layers = n_layers
        self.layer_list = []
        self.input_layer_list = []
        self.output_layer_list = []
        for node_no in sorted(layer_dict):
            if layer_dict[node_no]['layer_type'] != 'input':
                temp_list = [self.layer_list[layer_no] for layer_no in layer_dict[node_no]['incoming_layer_list']]
                layer_dict[node_no]['incoming_layer_list'] = temp_list
            temp_layer = Layer(**layer_dict[node_no])
            self.layer_list.append(temp_layer)
            if layer_dict[node_no]['layer_type'] == 'input':
                self.input_layer_list.append(temp_layer)
            elif layer_dict[node_no]['layer_type'] == 'output':
                self.output_layer_list.append(temp_layer)
                
    def prepare_mini_network(self):
        #x = self.input_layer_list[0].value
        #t = self.output_layer_list[0].value
        #EC = self.output_layer_list[0].loss
        w_list,b_list,act_list, act_name_list = [],[],[],[]
        ind_keep_next = None
        ind_keep_list = []
        for layer in self.layer_list:
            if layer.layer_type == 'input': continue
            if layer.drop_rate > 0:
                ind_keep = layer.rng.rand(layer.n_neuron) > layer.drop_rate
                ind_keep_list.append(ind_keep)
                b_list.append(layer.bias[ind_keep].copy())
                w_temp = layer.incoming_weight_list[0][:,ind_keep].copy()
                if ind_keep_next is not None:
                    w_temp = w_temp[ind_keep_next,:]
                ind_keep_next = ind_keep
                w_list.append(w_temp)
            else:
                if ind_keep_next is None:
                    w_list.append(layer.incoming_weight_list[0].copy())
                    b_list.append(layer.bias.copy())
                else:
                    b_list.append(layer.bias.copy())
                    w_list.append(layer.incoming_weight_list[0][ind_keep_next,:].copy())
                ind_keep_next = None
            act_list.append(layer.act_func)
            act_name_list.append(layer.act_func_name)
        return (w_list, b_list, act_list, act_name_list, ind_keep_list)
        
    def arange_dw_db(self, dw,db,ind_keep_list):
        dw_return, db_return = [],[]
        i,j = 0,0
        ind_keep_next = None
        for layer in self.layer_list:
            if layer.layer_type == 'input': continue
            dw_temp = np.zeros_like(layer.incoming_weight_list[0])
            db_temp = np.zeros_like(layer.bias)            
            if layer.drop_rate > 0:
                db_temp[ind_keep_list[j]] = db[i]
                if ind_keep_next is not None:
                    dw_temp[ind_keep_next,ind_keep_list[j]] = dw[i]
                else:
                    dw_temp[:,ind_keep_list[j]] = dw[i]
                ind_keep_next = ind_keep_list[j]
                j += 1
            else:
                db_temp = db[i]
                if ind_keep_next is not None:
                    dw_temp[ind_keep_next,:] = dw[i]
                    ind_keep_next = None
                else:
                    dw_temp = dw[i]
            i = i+1
            dw_return.append(dw_temp)
            db_return.append(db_temp)
        return (dw_return, db_return)
            
    
    def update_dw_db(self, dw,db,learning_rate=0.1, weight_decay=0, momentum=0, max_lr_iter = 5):
        i = 0
        for layer in self.layer_list:
            if layer.layer_type == 'input': continue
            layer.incoming_weight_list[0] = (1-weight_decay)*layer.incoming_weight_list[0] - dw[i]*learning_rate
            layer.bias = layer.bias - learning_rate*db[i]
            i = i + 1
    
    def transform(self, v=None):
        if v is None:
            return self.feed_forward()
        v = v if type(v) == type([]) else [v,]
        for i, layer in enumerate(self.input_layer_list):
            layer.value = v[i]
        for layer in self.layer_list:
            if layer.layer_type == 'input':
                continue
            layer.feed_forward(state = 'test')
        return [layer.value for layer in self.output_layer_list]
            
    def feed_forward(self):
        """
        returns output
        """
        for layer in self.layer_list:
            if layer.layer_type == 'input':
                continue
            layer.feed_forward()
        return [layer.value for layer in self.output_layer_list]
    
    def feed_and_back(self, isnorm = True):
        """
        returns grads
        """
        self.feed_forward()
        grads_list = []
        for layer in self.layer_list[::-1]:
            if layer.layer_type == 'input':
                continue
            if layer.layer_type == 'output':
                dy = np.ones_like(layer.value)
                layer.back_error = layer.value - layer.target
                if layer.loss == 'mse' and layer.act_func_name == 'sigmoid':
                    layer.back_error = layer.back_error*(layer.value*(1-layer.value))
            else:
                dy = None
            weight_grad_list, bias_grad, back_error_list = layer.back_prop_grads(dy, isnorm = isnorm)
            layer.weight_dir_list = weight_grad_list
            layer.bias_dir = bias_grad
            grads_list.append((weight_grad_list, bias_grad))
            for i, child_layer in enumerate(layer.incoming_layer_list):
                if child_layer.layer_type == 'hidden':
                    child_layer.back_error += back_error_list[i]
        return grads_list
        
    def empirical_error(self, target = None):
        if target is not None:
            target = target if type(target) == type([]) else [target,]
            for layer, target_ in zip(self.output_layer_list, target):
                layer.target = target_ 
        return sum([layer.layer_error() for layer in self.output_layer_list])
        
    def generate_batch(self, batch_size):
        n_samples = self.input_layer_list[0].link2input.shape[0]
        if batch_size > n_samples:
            for inp_layer in self.input_layer_list:
                inp_layer.value = inp_layer.link2input
            for out_layer in self.output_layer_list:
                out_layer.target = out_layer.link2target
            yield 0
        else:
            beg,last,batch_no = 0,0,0
            while last != n_samples:
                last = np.min([n_samples,beg+batch_size])
                for inp_layer in self.input_layer_list:
                    inp_layer.value = inp_layer.link2input[beg:last]
                for out_layer in self.output_layer_list:
                    out_layer.target = out_layer.link2target[beg:last]
                yield batch_no
                batch_no += 1
                beg = last
                
    def _fit_with_minimize(self, learning_rate=0.1, weight_decay=0, momentum=0, verbose = True, max_lr_iter = 5, isnorm = True):
        big_weight = weight_extend(self)
        big_weight, _,_ = minimize.minimize(big_weight, helper_func_eval, (self, isnorm), maxnumlinesearch=3, verbose = False)
        weight_compress(big_weight, self)
        if verbose:
            self.feed_forward()
            return self.empirical_error()
        
    
    def _fit(self, learning_rate=0.1, weight_decay=0, momentum=0, max_lr_iter = 5, isnorm = True, verbose = True):
        
        self.feed_forward()
        #error = self.empirical_error()
        
        for layer in self.layer_list[::-1]:
            if layer.layer_type == 'input':
                continue
            if layer.layer_type == 'output':
                dy = np.ones_like(layer.value)
                layer.back_error = layer.value - layer.target
                if layer.loss == 'mse' and layer.act_func_name == 'sigmoid':
                    layer.back_error = layer.back_error*(layer.value*(1-layer.value))
            else:
                dy = None
            weight_grad_list, bias_grad, back_error_list = layer.back_prop_grads(dy, isnorm = isnorm)
            #backup_weight_list = layer.incoming_weight_list
            #backup_bias = layer.bias
            #backup_weight_grad_list = layer.weight_grad_list
            #backup_bias_grad = layer.bias_grad
            #lr_iter = 1
            #while True:
            layer.update_weight_bias(weight_grad_list, bias_grad, learning_rate, weight_decay, momentum)
                #self.feed_forward()
                #if self.empirical_error() < error or max_lr_iter < lr_iter: break
                #lr_iter += 1
                #learning_rate /= 2
                #layer.incoming_weight_list = backup_weight_list
                #layer.bias = backup_bias
                #layer.weight_grad_list = backup_weight_grad_list
                #layer.bias_grad = backup_bias_grad
            #if lr_iter > max_lr_iter:
            #    print("increase the max_lr_iter")
            
            for i, child_layer in enumerate(layer.incoming_layer_list):
                if child_layer.layer_type == 'hidden':
                    child_layer.back_error += back_error_list[i]
            
            
        if verbose:
            return self.empirical_error()
            
    def _fit_dropout(self,learning_rate=0.1, weight_decay=0, momentum=0, max_lr_iter = 5, isnorm = True, verbose = True):
        #self.feed_forward()        
        X_temp = self.input_layer_list[0].value
        Y_temp = self.output_layer_list[0].target
        EC = self.output_layer_list[0].loss
        pl = 0
        dwt,dbt = [0]*(len(self.layer_list)-1), [0]*(len(self.layer_list)-1)
        for x, y in zip(X_temp, Y_temp):
            w_list, b_list, act_list, act_name_list, ind_keep_list = self.prepare_mini_network()
            dw,db,error = miniNetwork(x, y, w_list, b_list, act_list, act_name_list, EC, isnorm = isnorm)
            dw,db = self.arange_dw_db(dw,db,ind_keep_list)
            #self.update_dw_db(dw,db,learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum,max_lr_iter = max_lr_iter)
            pl += error
            for i,(dw_,db_) in enumerate(zip(dw,db)):
                dwt[i] += dw_
                dbt[i] += db_
        if isnorm:
            for i,(dwt_,dbt_) in enumerate(zip(dwt,dbt)):
                dwt[i] /= np.linalg.norm(dwt_)
                dbt[i] /= np.linalg.norm(dbt_)
        self.update_dw_db(dwt,dbt,learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum,max_lr_iter = max_lr_iter)
        return pl
    
    def fit(self, batch_size=100, learning_rate=None, weight_decay=None, momentum=None, 
            n_iter=100, verbose = True, random_state=None, switch_point = None, isnorm = True, only_dropout = False):
        
        learning_rate = step_iterator(0.1,0.1,0) if learning_rate is None else learning_rate
        weight_decay = step_iterator(0,0,0) if weight_decay is None else weight_decay
        momentum = step_iterator(0,0,0.) if momentum is None else momentum
        switch_point = n_iter if switch_point is None else switch_point        
        
        rng = check_random_state(random_state)
        for layer in self.layer_list:
            layer.assign_random_state(rng)
            
        fit_func = self._fit
        for i_outer, iteration in enumerate(xrange(n_iter)):
            pl = 0.
            lr = learning_rate.next()
            wd = weight_decay.next()
            mom = momentum.next()
#            r_index = np.arange(self.input_layer_list[0].link2input.shape[0])
#            np.random.shuffle(r_index)
#            for inp_layer in self.input_layer_list:
#                inp_layer.link2input = inp_layer.link2input[r_index]
#            for out_layer in self.output_layer_list:
#                out_layer.link2target = out_layer.link2target[r_index]
            if verbose:
                begin = time()
            
            fit_func = self._fit_dropout if only_dropout else fit_func
            for batch_no in self.generate_batch(batch_size):
                #print("batch no:%d"%batch_no)
                pl_batch = fit_func(learning_rate = lr, weight_decay = wd, momentum = mom, max_lr_iter = 5, isnorm = isnorm)
                if verbose:
                    pl += pl_batch.sum()
            fit_func = self._fit if i_outer<switch_point else self._fit_with_minimize
            if verbose:
                #pl /= n_samples
                end = time()
                print("Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (iteration, pl, end-begin))
        return self
        
if __name__ == '__main__':
    X1, X2 = np.random.randn(100,10), np.random.randn(100,5)
    Y1,Y2 = np.random.randn(100,2), np.random.multinomial(1,pvals=[0.2,0.5,0.3],size=100)
    W1, W2 = np.random.randn(10,20), np.random.randn(5,20)
    W3, W4 = np.random.randn(20,2), np.random.randn(20,3)
    def layer_test():
        l1 = Layer(n_neuron = 10, value = X1, layer_type = 'input', link2input = X1)
        l2 = Layer(n_neuron = 5, value = X2, layer_type = 'input', link2input = X2)
        l3 = Layer(n_neuron = 20, incoming_layer_list = [l1, l2], incoming_weight_list = [W1,W2], bias = np.random.randn(20),
                   act_func_name = 'sigmoid', layer_type = 'hidden')
        l4 = Layer(n_neuron = 2, incoming_layer_list = [l3,], incoming_weight_list = [], bias = np.random.randn(2),
                   act_func_name = 'linear', layer_type = 'output', link2target = Y1, loss = 'mse', back_error = np.random.randn(100,2))
        l5 = Layer(n_neuron = 3, incoming_layer_list = [l3,], incoming_weight_list = [W4,], bias = np.random.randn(3),
                   act_func_name = 'softmax', layer_type = 'output', link2target = Y2,  back_error = np.random.randn(100,3))
    
        l3.feed_forward()
        l4.feed_forward(); l5.feed_forward(); l4.target = l4.link2target; l5.target = l5.link2target;
        l4.layer_error()
        l5.layer_error()
        l4.back_prop_grads()
        (dw,db,back_error) = l5.back_prop_grads(dy = np.ones((100,3)))
        l5.update_weight_bias(dw,db)
        dw,db,back_error = l3.back_prop_grads()
        l3.update_weight_bias(dw,db)
        
    #layer_test()
    def network_test():
        layer_dict = {'0':
            {'n_neuron': 10,        
             'incoming_layer_list': [],
             'incoming_weight_list': [],
             'bias': None,
             'loss': 'cross_entropy',
             'act_func_name': 'sigmoid',
             'value': None,
             'layer_type': 'input',
             'back_error' : 0,
             'drop_rate': 0.0,
             'link2input': X1,
             'link2target': None
             },
             '1': {'n_neuron':5, 'layer_type':'input', 'link2input':X2},
             '2': {'n_neuron': 20, 'incoming_layer_list': [0,1], 'act_func_name':'sigmoid','layer_type':'hidden', 'drop_rate':0.2},
             '3': {'n_neuron':2, 'incoming_layer_list': [2,], 'loss':'mse', 'act_func_name':'linear',
                   'layer_type':'output', 'link2target':Y1},
             '4': {'n_neuron':3, 'incoming_layer_list': [2,], 'act_func_name':'softmax',
                   'layer_type':'output', 'link2target':Y2}
            }
        network = NeuralNetwork(5,layer_dict = layer_dict)
        #outputs = network.feed_forward()
        #error = network.empirical_error()
        return network.fit(batch_size = 20, learning_rate = step_iterator(0.1,0.01,0), isnorm = True,
                           weight_decay = step_iterator(1e-5,1e-5,0), n_iter = 500, random_state = 20, switch_point = 500)
        
    network = network_test()
    print(network.empirical_error(target = network.transform(v = [X1, X2])))
    print(np.linalg.norm(network.layer_list[2].incoming_weight_list[0]))
    print(np.linalg.norm(network.layer_list[2].incoming_weight_list[1]))
    #print(network.layer_list[2].incoming_weight_list[0])