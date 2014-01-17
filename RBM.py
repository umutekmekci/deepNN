# -*- coding: utf-8 -*-
"""
Created on Wed Nov 06 20:05:38 2013

@author: daredavil
"""

#Restricted Boltzman Machines with PCD-k learning
from __future__ import print_function
from __future__ import division
import numpy as np
from time import time
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import logistic_sigmoid

#def logistic_sigmoid(X):
#    return 1/(1+np.exp(-X))

def linear_act(X):
    return X
    
def relu_act(X):
    r_ = X.copy()
    r_[X<0] = 0
    return r_

def softmax_act(X):
    if X.ndim == 1:
        return np.exp(X)/np.exp(X).sum()
    return np.exp(X)/np.exp(X).sum(axis=1)[:,np.newaxis]

#def logistic_sigmoid(X):
#    return 1/(1 + np.exp(-X))
            
def step_iterator(beg, last, step):
    if step == 0:
        while True:
            yield beg
    f1 = min if np.sign(step) == -1 else max
    f2 = max if np.sign(step) == -1 else min
    step = f1(-step,step)
    while True:
        if beg == last:
            yield beg
        else:
            yield beg
            beg += step
        beg = f2(beg, last)

def batch_func_generator(X_list, batch_size = 100):
    X_list = X_list if type(X_list) == type([]) else [X_list,]
#    org = X_list[1].copy()
    def batch_func(layer_list):
        n_samples = X_list[0].shape[0]
        if batch_size > n_samples:
            for i, layer in enumerate(layer_list):
                layer.value = X_list[i]
            yield n_samples
        else:
 #           X_list[1] = org.copy()        
 #           r_inds = np.random.choice(n_samples, size=np.round((n_samples*3)/10).astype('int') ,replace = False).astype('int')
 #           X_list[1][r_inds,:] = 0
           # r_index = np.arange(n_samples)
           # np.random.shuffle(r_index)
           # for i, x in enumerate(X_list):
           #     X_list[i] = x[r_index]
            beg,last = 0,0
            while last != n_samples:
                last = np.min([n_samples,beg+batch_size])
                for i, layer in enumerate(layer_list):
                    layer.value = X_list[i][beg:last]
                yield layer.value.shape[0]
                beg = last
    return batch_func
        
        
class UnknownName(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        
class RBMLayer:
    def __init__(self, dimension, value = None, bias = None):
        self.dimension = dimension
        self.value = value
        self.bias = np.zeros(dimension) if bias is None else bias 
        self.bias_grad = 0.0
        
class BinaryLayer(RBMLayer):
    def __init__(self, dimension, value = None, bias = None):
        RBMLayer.__init__(self, dimension, value = value, bias = bias)
        self.act_func = logistic_sigmoid
        self.act_func_name = 'sigmoid'
        
    def sample(self, rng, act = None, assign = False):
        act = self.value if act is None else act
        index = rng.uniform(size=act.size) < act
        h = np.zeros_like(act)
        h[index] = 1.0
        if assign:
            self.value = h
        return h
        
class LinearLayer(RBMLayer):
     def __init__(self, dimension, value = None, bias = None, sigma=1):
        RBMLayer.__init__(self, dimension, value = value, bias = bias)
        self.sigma = sigma
        self.act_func = linear_act
        self.act_func_name = 'linear'
        
     def sample(self, rng, act = None, assign = False):
         act = self.value if act is None else act
         h = act + rng.normal(0,1,act.shape)*self.sigma
         if assign:
             self.value = h
         return h
         
class ReluLayer(RBMLayer):
    def __init__(self, dimension, value = None, bias = None):
        RBMLayer.__init__(self, dimension, value = value, bias = bias)
        self.act_func = relu_act
        self.act_func_name = 'relu'
    
    def sample(self, rng, act = None, assign = False):
         act = self.value if act is None else act
         h = act + rng.normal(0,1,act.shape)*np.sqrt(logistic_sigmoid(act))
         h[h<0] = 0
         if assign:
             self.value = h
         return h
         
class SoftmaxLayer(RBMLayer):
    def __init__(self, dimension, value = None, bias = None):
        RBMLayer.__init__(self, dimension, value = value, bias = bias)
        self.act_func = softmax_act
        self.act_func_name = 'softmax'
        
    def sample(self, rng, act = None, D = 1, assign = False):
        act = self.value if act is None else act
        D = np.ones(act.shape[0])*D if np.isscalar(D) else D
        if D.shape[0] != act.shape[0]:
            raise UnknownName("Sampling error at softmax layer")
        h = np.zeros_like(act)
        for i, d in enumerate(D):
            h[i] = rng.multinomial(d, act[i], size=1)
        if assign:
            self.value = h
        return h
     


class BinRBM:
    """
    layers_dict = {layer_no:
                            {'value':, 'bias', 'sigma', 'layer_type':, 'layer_name'}}
    layer_type: binary, linear, softmax
    layer_name: input, hidden
    """
    
    def __init__(self, layers_dict, weight_list = None, random_state = None):
        self.input_layer_list = []
        for layer_no in sorted(layers_dict):
            if layers_dict[layer_no]['layer_name'] == 'input':
                if layers_dict[layer_no]['layer_type'] == 'binary':
                    layer = BinaryLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                        bias = layers_dict[layer_no].get('bias', None))
                    self.input_layer_list.append(layer)
                elif layers_dict[layer_no]['layer_type'] == 'linear':
                    layer = LinearLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                        bias = layers_dict[layer_no].get('bias', None),
                                        sigma = layers_dict[layer_no].get('sigma', 1))
                    self.input_layer_list.append(layer)
                elif layers_dict[layer_no]['layer_type'] == 'softmax':
                    layer = SoftmaxLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                         bias = layers_dict[layer_no].get('bias', None))
                    self.input_layer_list.append(layer)
                elif layers_dict[layer_no]['layer_type'] == 'relu':
                    layer = ReluLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                         bias = layers_dict[layer_no].get('bias', None))
                    self.input_layer_list.append(layer)
            else:
                if layers_dict[layer_no]['layer_type'] == 'binary':
                    self.hidden_layer = BinaryLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                                    value = layers_dict[layer_no].get('value',None),
                                                    bias = layers_dict[layer_no].get('bias', None))
                elif layers_dict[layer_no]['layer_type'] == 'linear':
                    self.hidden_layer = LinearLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                        bias = layers_dict[layer_no].get('bias', None),
                                        sigma = layers_dict[layer_no].get('sigma', 1))
                elif layers_dict[layer_no]['layer_type'] == 'softmax':
                    self.hidden_layer = SoftmaxLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                         bias = layers_dict[layer_no].get('bias', None))
                elif layers_dict[layer_no]['layer_type'] == 'relu':
                    self.hidden_layer = ReluLayer(dimension = layers_dict[layer_no].get('dimension', None),
                                        value = layers_dict[layer_no].get('value',None),
                                         bias = layers_dict[layer_no].get('bias', None))
        self.rng = check_random_state(random_state)
        self.random_state = random_state
        self.weight_list = weight_list
        if self.weight_list is None:
            self._init_weight_list()
        self.weight_grad_list = [0.0,]*(len(self.weight_list))
            
    def _init_weight_list(self):
        weight_list = []
        for layer in self.input_layer_list:
            weight = self.rng.normal(0,0.01,(self.hidden_layer.dimension, layer.dimension))
            weight_list.append(weight)
        self.weight_list = weight_list
            
            
    def feed_forward(self, sample = False):
        act = 0
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            sigma = layer.sigma if hasattr(layer,'sigma') else 1
            act += np.dot(layer.value/sigma, weight.T)
        act += self.hidden_layer.bias
        self.hidden_layer.value = self.hidden_layer.act_func(act)
        if sample:
            self.hidden_layer.value = self.hidden_layer.sample(self.rng)
        
    def transform(self, V, sample = False):
        V = V if type(V) == type([]) else [V,]
        act = 0
        for v, layer, weight in zip(V, self.input_layer_list, self.weight_list):
            sigma = layer.sigma if hasattr(layer,'sigma') else 1
            act += np.dot(v/sigma, weight.T)
        act += self.hidden_layer.bias
        act = self.hidden_layer.act_func(act)
        if sample:
            act = self.hidden_layer.sample(self.rng, act)
        return act
            
    
    def feed_back(self, sample = False):
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            sigma = layer.sigma if hasattr(layer,'sigma') else 1
            layer.value = layer.act_func(np.dot(self.hidden_layer.value, weight)*sigma + layer.bias)
            if sample:
                layer.value = layer.sample(self.rng)
        
        
    def free_energy(self,v):
        """Computes the free energy F(v) = -log sum_h [exp(-E(v,h))]
           minus Unnormalized marginal likelihood
           -log P(v;Q) = log Z - b'v - sum_j [log(1 + exp(a_j + W_jv))]
           j=1:n_components, a_j = intercept_hidden_, b=intercept_visible
           W = components_
        
        returns
        -------
        v : should be augmented if there are two or more visible layers
        free_energy : array-like shape (n_samples,)
            The value of the free energy
        """
        t = 0
        for layer in self.input_layer_list:
            t += layer.dimension
        if v.shape[1] != t:
            raise UnknownName("v should be augmented")
        weight = np.hstack(self.weight_list)
        bias = np.hstack([layer.bias for layer in self.input_layer_list])
        fe = -np.dot(v, bias) - np.log(1 + \
        np.exp(np.dot(v,weight.T) + self.hidden_layer.bias)).sum(axis=1)
        return fe
        
        
    def run_gibbs(self, X = None, step_size = 1, sample = True):
        """Perform step_size gibbs sampling step
            and update states of RBM
        sample : Bool, default:True
            if true iterate on states else probabilities
        returns
        -------
        None
        """
        if X is not None:
            X = X if type(X) == type([]) else [X,]
            for layer, x in zip(self.input_layer_list, X):
                layer.value = x
            
        for _ in xrange(step_size):
            self.feed_forward(sample)
            self.feed_back(sample)
        
    def fix_any_source_and_run_gibs(self, X = None, fixed_layer_list = None, step_size = 1, sample=True):
        """
        fix_layer_list: list
            identity of sources which are fixed, eg. [0, ], source 0 is fixed
        """
        if X is not None:
            X = X if type(X) == type([]) else [X,]
            for layer, x in zip(self.input_layer_list, X):
                layer.value = x
        fixed_layer_list = fixed_layer_list if type(fixed_layer_list)==type([]) else [fixed_layer_list,]
        cached = []
        for layer_no in fixed_layer_list:
            cached.append(self.input_layer_list[layer_no].value.copy())
        for _ in xrange(step_size):
            self.feed_forward(sample)
            self.feed_back(sample)
            for i, layer_no in enumerate(fixed_layer_list):
                self.input_layer_list[layer_no].value = cached[i]
        
        
    def _fit(self, batch_size, PCD = True, error_function = "pl", learning_rate = 0.1, weight_decay = 0,
             momentum = 0, k = 1, sparsity_cond = False, sparsity_target = 0.1, sparsity_lambda = 0.1, verbose = True):
        """ Inner fit for one mini-batch
        
        returns
        -------
        pseudo_likelihood : array-like shape (n_samples,)
            If verbose=True, pseudo-likelihood estimates for this batch
        """
        self.feed_forward()
        hidden_pos_sum = self.hidden_layer.value.sum(axis = 0)
        pos_corr_list = []
        vis_pos_sum_list = []
        cached_vis = []
        for layer in self.input_layer_list:
            pos_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_pos_sum_list.append(layer.value.sum(axis = 0))
            if verbose:
                cached_vis.append(layer.value.copy())
            
        if sparsity_cond:
            sparse_hidden_bias = (sparsity_target - hidden_pos_sum) * (self.hidden_layer.value*(1-self.hidden_layer.value)).mean(axis = 0)
            #sparse_hidden_bias /= np.linalg.norm(sparse_hidden_bias)
            sparse_hidden_bias = 2*sparsity_lambda*sparse_hidden_bias
        else:
            sparse_hidden_bias = 0
        
        #CD iteration
        if PCD:
            self.hidden_layer.value = self.h_samples_
        for _ in xrange(k):
            self.feed_back()
            self.feed_forward()
        hidden_neg_sum = self.hidden_layer.value.sum(axis = 0)
        neg_corr_list = []
        vis_neg_sum_list = []
        for layer in self.input_layer_list:
            neg_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_neg_sum_list.append(layer.value.sum(axis = 0))
        
        #update parameters
       
        lr = float(learning_rate)/batch_size # to average the gradient
        for i, (weight_grad, layer) in enumerate(zip(self.weight_grad_list, self.input_layer_list)):
            sigma = layer.sigma if hasattr(layer,'sigma') else 1
            self.weight_grad_list[i] = lr*(pos_corr_list[i] - neg_corr_list[i])*(1/sigma) + momentum*weight_grad - weight_decay*self.weight_list[i]
  #          self.weight_grad_list[i] /= np.linalg.norm(self.weight_grad_list[i])            
            self.weight_list[i] += self.weight_grad_list[i]
            layer.bias_grad = lr*(vis_pos_sum_list[i] - vis_neg_sum_list[i]) * (1/(sigma**2)) + momentum*layer.bias_grad
 #           layer.bias_grad /= np.linalg.norm(layer.bias_grad)            
            layer.bias += layer.bias_grad
        self.hidden_layer.bias_grad = lr*(hidden_pos_sum - hidden_neg_sum) + momentum*self.hidden_layer.bias_grad
#        self.hidden_layer.bias_grad /= np.linalg.norm(self.hidden_layer.bias_grad)        
        self.hidden_layer.bias += (self.hidden_layer.bias_grad + sparse_hidden_bias)
        
        if PCD:
            self.h_samples_ = self.hidden_layer.value
        
        if verbose:
            if error_function == "pl":
                return self.score_samples(cached_vis)
            else:
                error = 0
                sat = np.min((cached_vis[0].shape[0], self.input_layer_list[0].value.shape[0]))
                sut = np.min((cached_vis[0].shape[1], self.input_layer_list[0].value.shape[1]))
                for i, layer in enumerate(self.input_layer_list):
                    error += np.sum((cached_vis[i] - layer.value[:sat,:sut])**2)
                return error
            
    def score_samples(self,v):
        """Computer the pseudo-likelihood of v
        
        returns
        --------
        pseudo_likelihood : array-like shape (n_samples,)
        """
        v = np.hstack(v)
        fe = self.free_energy(v)
        
        v_ = v.copy()
        rng = check_random_state(self.random_state)
        i_ = rng.randint(0,v.shape[1],v.shape[0])
        v_[np.arange(v.shape[0]),i_] = 1 - v_[np.arange(v.shape[0]), i_]
        fe_ = self.free_energy(v_)
        
        return v.shape[1] * logistic_sigmoid(fe_ - fe, log=True)
            
        
    def fit(self, batch_func, PCD = True, error_function="pl", learning_rate = None,
            weight_decay = None, momentum = None, k = 1, perst_size = 100, n_iter = 10,
            sparsity_cond = False, sparsity_target = 0.1, sparsity_lambda = 0.1, verbose = True):
        """Fit the model to the data X
        
        Parameters
        ----------
        X : array-like shape (n_samples, n_features)
            Training data
        
        returns
        -------
        self
        """
        #X, = check_arrays(X, sparse_format='csc', dtype=np.float)
        learning_rate = step_iterator(0.1,0.1,0) if learning_rate is None else learning_rate
        weight_decay = step_iterator(0,0,0) if weight_decay is None else weight_decay
        momentum = step_iterator(0,0,0.) if momentum is None else momentum
        
        self.h_samples_ = np.zeros((perst_size,self.hidden_layer.dimension))
        
        for i_outer, iteration in enumerate(xrange(n_iter)):
            pl = 0.
            lr = learning_rate.next()
            wd = weight_decay.next()
            mom = momentum.next()
            #if i_outer > 20:
            #    k = 5
            if verbose:
                begin = time()
            
            n_total = 0
            for i_inner, batch_size in enumerate(batch_func(self.input_layer_list)):
                pl_batch = self._fit(batch_size, PCD = PCD, error_function = error_function, 
                                     learning_rate = lr, weight_decay = wd, momentum = mom, k=k,
                                     sparsity_cond = sparsity_cond, sparsity_target = sparsity_target, sparsity_lambda = sparsity_lambda)
                if verbose:
                    pl += pl_batch.sum()
                    if i_inner == 20:
                        pass
                n_total += batch_size
            
            if verbose:
                pl /= n_total
                end = time()
                print("Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (iteration, pl, end-begin))
        return self
        
class RSMRBM(BinRBM):
    def __init__(self, layers_dict, weight_list = None, random_state = None):
        BinRBM.__init__(self, layers_dict = layers_dict, weight_list = weight_list, random_state=random_state)
    
    def feed_forward(self, D, sample = False):
        act = 0
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            act += np.dot(layer.value, weight.T)
        act += np.outer(D,self.hidden_layer.bias)
        self.hidden_layer.value = self.hidden_layer.act_func(act)
        if sample:
            self.hidden_layer.value = self.hidden_layer.sample(self.rng)
            
    def feed_back(self, D):
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            layer.value = layer.act_func(np.dot(self.hidden_layer.value, weight) + layer.bias)
            layer.sample(self.rng, D=D, assign=True)
            
    def transform(self, V, sample = False):
        V = V if type(V) == type([]) else [V,]
        act = 0
        for v, weight in zip(V, self.weight_list):
            act += np.dot(v, weight.T)
        act += np.outer(V[0].sum(axis=1),self.hidden_layer.bias)
        act = self.hidden_layer.act_func(act)
        if sample:
            act = self.hidden_layer.sample(self.rng, act)
        return act
        
    def perplexity(self, v_train, v_test):
        v_train = v_train if type(v_train) == type([]) else [v_train,]
        D = v_train.sum(aixs = 1)
        for i,layer in enumerate(self.input_layer_list):
            layer.value = v_train[i]
        self.feed_forward(D)
        ppl = 0
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            layer.value = layer.act_func(np.dot(self.hidden_layer.value, weight) + layer.bias)
            ppl += np.nansum(v_test*np.log(layer.value))
        all_word_sum = np.sum(v_test)
        ppl = np.exp(-ppl/all_word_sum)
        return ppl
        
            
    def _fit(self, batch_size, PCD = True, error_function = "pl", learning_rate = 0.1, weight_decay = 0,
             momentum = 0, k = 1, sparsity_cond = False, sparsity_target = 0.1, sparsity_lambda = 0.1, verbose = True):
        """ Inner fit for one mini-batch
        
        returns
        -------
        pseudo_likelihood : array-like shape (n_samples,)
            If verbose=True, pseudo-likelihood estimates for this batch
        """
        D = self.input_layer_list[0].value.sum(axis=1)
        self.feed_forward(D)
        hidden_pos_sum = self.hidden_layer.value.sum(axis = 0)
        pos_corr_list = []
        vis_pos_sum_list = []
        cached_vis = []
        for layer in self.input_layer_list:
            pos_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_pos_sum_list.append(layer.value.sum(axis = 0))
            if verbose:
                cached_vis.append(layer.value.copy())
            
        if sparsity_cond:
            sparse_hidden_bias = (sparsity_target - hidden_pos_sum) * (self.hidden_layer.value*(1-self.hidden_layer.value)).mean(axis = 0)
            sparse_hidden_bias = 2*sparsity_lambda*sparse_hidden_bias
        else:
            sparse_hidden_bias = 0
        
        #CD iteration
        if PCD:
            self.hidden_layer.value = self.h_samples_
        for _ in xrange(k):
            self.feed_back(D)
            self.feed_forward(D)
        hidden_neg_sum = self.hidden_layer.value.sum(axis = 0)
        neg_corr_list = []
        vis_neg_sum_list = []
        for layer in self.input_layer_list:
            neg_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_neg_sum_list.append(layer.value.sum(axis = 0))
        
        #update parameters
       
        lr = float(learning_rate)#/batch_size # to average the gradient
        for i, (weight_grad, layer) in enumerate(zip(self.weight_grad_list, self.input_layer_list)):
            diff_corr = (pos_corr_list[i] - neg_corr_list[i])#/batch_size
            diff_corr /= np.linalg.norm(diff_corr)
            self.weight_grad_list[i] = diff_corr*lr
            self.weight_list[i] = (1-weight_decay)*self.weight_list[i] + self.weight_grad_list[i] + momentum*weight_grad
            diff_bias_sum = (vis_pos_sum_list[i] - vis_neg_sum_list[i])#/batch_size
            diff_bias_sum /= np.linalg.norm(diff_bias_sum)
            layer.bias_grad = lr*diff_bias_sum + momentum*layer.bias_grad
            layer.bias += layer.bias_grad
        diff_hidden_sum = (hidden_pos_sum - hidden_neg_sum)#/batch_size
        diff_hidden_sum /= np.linalg.norm(diff_hidden_sum)
        self.hidden_layer.bias_grad = lr*diff_hidden_sum + momentum*self.hidden_layer.bias_grad
        self.hidden_layer.bias += (self.hidden_layer.bias_grad + sparse_hidden_bias)
        
        if PCD:
            self.h_samples_ = self.hidden_layer.value
        
        if verbose:
            if error_function == "pl":
                return self.score_samples(cached_vis)
            else:
                error = 0
                sat = np.min((cached_vis[0].shape[0], self.input_layer_list[0].value.shape[0]))
                sut = np.min((cached_vis[0].shape[1], self.input_layer_list[0].value.shape[1]))
                for i, layer in enumerate(self.input_layer_list):
                    error += np.sum((cached_vis[i] - layer.value[:sat,:sut])**2)
                return error
                
    def run_gibbs(self, X = None, step_size = 1, sample = False):
        """Perform step_size gibbs sampling step
            and update states of RBM
        sample : Bool, default:True
            if true iterate on states else probabilities
        returns
        -------
        None
        """
        if X is not None:
            X = X if type(X) == type([]) else [X,]
            for layer, x in zip(self.input_layer_list, X):
                layer.value = x
        D = self.input_layer_list[0].value.sum(axis =1)
        for _ in xrange(step_size):
            self.feed_forward(D,sample)
            self.feed_back(D)
            
    def fix_any_source_and_run_gibs(self, X = None, fixed_layer_list = None, step_size = 1, sample=True):
        """
        fix_layer_list: list
            identity of sources which are fixed, eg. [0, ], source 0 is fixed
        """
        if X is not None:
            X = X if type(X) == type([]) else [X,]
            for layer, x in zip(self.input_layer_list, X):
                layer.value = x
        D = self.input_layer_list[0].value.sum(axis =1)
        fixed_layer_list = fixed_layer_list if type(fixed_layer_list)==type([]) else [fixed_layer_list,]
        cached = []
        for layer_no in fixed_layer_list:
            cached.append(self.input_layer_list[layer_no].value.copy())
        for _ in xrange(step_size):
            self.feed_forward(D, sample)
            self.feed_back(D)
            for i, layer_no in enumerate(fixed_layer_list):
                self.input_layer_list[layer_no].value = cached[i]
                
class GaussRBM(BinRBM):
    def __init__(self, layers_dict, weight_list = None, random_state = None):
        BinRBM.__init__(self, layers_dict = layers_dict, weight_list = weight_list, random_state=random_state)
        
    def feed_forward(self, sample = False):
        act = 0
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            sigma = layer.sigma
            act += np.dot(layer.value/sigma, weight.T)
        act += self.hidden_layer.bias
        if np.any(np.isnan(act)) or np.any(np.isinf(act)):
            pass
        self.hidden_layer.value = self.hidden_layer.act_func(act)
        if sample:
            self.hidden_layer.value = self.hidden_layer.sample(self.rng)
            
    def transform(self, V, sample = False):
        V = V if type(V) == type([]) else [V,]
        act = 0
        for v, layer, weight in zip(V, self.input_layer_list, self.weight_list):
            sigma = layer.sigma
            act += np.dot(v/sigma, weight.T)
        act += self.hidden_layer.bias
        act = self.hidden_layer.act_func(act)
        if sample:
            act = self.hidden_layer.sample(self.rng, act)
        return act
            
    
    def feed_back(self, sample = False):
        for layer, weight in zip(self.input_layer_list, self.weight_list):
            sigma = layer.sigma
            layer.value = layer.act_func(np.dot(self.hidden_layer.value, weight)*sigma + layer.bias)
            if sample:
                layer.value = layer.sample(self.rng)
                
    def _fit(self, batch_size, PCD = True, error_function = "pl", learning_rate = 0.1, weight_decay = 0,
             momentum = 0, k = 1, sparsity_cond = False, sparsity_target = 0.1, sparsity_lambda = 0.1, verbose = True):
        """ Inner fit for one mini-batch
        
        returns
        -------
        pseudo_likelihood : array-like shape (n_samples,)
            If verbose=True, pseudo-likelihood estimates for this batch
        """
        self.feed_forward()
        hidden_pos_sum = self.hidden_layer.value.sum(axis = 0)
        pos_corr_list = []
        vis_pos_sum_list = []
        cached_vis = []
        for layer in self.input_layer_list:
            pos_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_pos_sum_list.append(layer.value.sum(axis = 0))
            if verbose:
                cached_vis.append(layer.value.copy())
            
        if sparsity_cond:
            sparse_hidden_bias = (sparsity_target - hidden_pos_sum) * (self.hidden_layer.value*(1-self.hidden_layer.value)).mean(axis = 0)*2*sparsity_lambda
            #sparse_hidden_bias /= np.linalg.norm(sparse_hidden_bias)
            #sparse_hidden_bias = 2*sparsity_lambda*sparse_hidden_bias 
        else:
            sparse_hidden_bias = 0
        
        #CD iteration
        if PCD:
            self.hidden_layer.value = self.h_samples_
        for _ in xrange(k):
            self.feed_back()
            self.feed_forward()
        hidden_neg_sum = self.hidden_layer.value.sum(axis = 0)
        neg_corr_list = []
        vis_neg_sum_list = []
        for layer in self.input_layer_list:
            neg_corr_list.append(np.dot(self.hidden_layer.value.T, layer.value))
            vis_neg_sum_list.append(layer.value.sum(axis = 0))
        
        #update parameters
       
        lr = float(learning_rate)#/batch_size # to average the gradient
        for i, (weight_grad, layer) in enumerate(zip(self.weight_grad_list, self.input_layer_list)):
            sigma = layer.sigma
            self.weight_grad_list[i] = (pos_corr_list[i] - neg_corr_list[i])*(1/sigma) 
            self.weight_grad_list[i] /= np.linalg.norm(self.weight_grad_list[i])  
            self.weight_grad_list[i] *= lr
            self.weight_list[i] = (1-weight_decay)*self.weight_list[i] + self.weight_grad_list[i] + momentum*weight_grad
            layer.bias_grad = (vis_pos_sum_list[i] - vis_neg_sum_list[i]) * (1/(sigma**2))
            layer.bias_grad /= np.linalg.norm(layer.bias_grad) 
            layer.bias_grad *= lr            
            layer.bias += (layer.bias_grad + momentum*layer.bias_grad)
        self.hidden_layer.bias_grad = (hidden_pos_sum - hidden_neg_sum)
        self.hidden_layer.bias_grad /= np.linalg.norm(self.hidden_layer.bias_grad)        
        self.hidden_layer.bias_grad *= lr            
        self.hidden_layer.bias += (self.hidden_layer.bias_grad + sparse_hidden_bias + momentum*self.hidden_layer.bias_grad)
        
        if PCD:
            self.h_samples_ = self.hidden_layer.value
        
        if verbose:
            if error_function == "pl":
                return self.score_samples(cached_vis)
            else:
                error = 0
                sat = np.min((cached_vis[0].shape[0], self.input_layer_list[0].value.shape[0]))
                sut = np.min((cached_vis[0].shape[1], self.input_layer_list[0].value.shape[1]))
                for i, layer in enumerate(self.input_layer_list):
                    error += np.sum((cached_vis[i] - layer.value[:sat,:sut])**2)
                return error
            
    
        
if __name__ == "__main__":
    
    def testRBM():
        X = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
        learning_rate = step_iterator(beg = 0.001, last = 0, step = 0)
        weight_decay = step_iterator(beg = 1e-5, last = 1e-3, step = 1e-5)
        momentum = step_iterator(beg = 0.5, last = 0.9, step = 0.01)
        user = np.array([[0,0,0,1,1,0]])
        layers_dict = {0: {
                            'dimension':6,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':2,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
        batch_func = batch_func_generator(X, batch_size = 10)
        rbm = BinRBM(layers_dict, weight_list = None, random_state = 20)
        rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 10, n_iter = 100, verbose = True)
        print(rbm.transform(V = user))
        rbm.run_gibbs(X = user, step_size = 100, sample = False)
        print(rbm.hidden_layer.value)
    #testRBM()
    def testRSMRBM():
        import fmatrix
        X = fmatrix.parse('train')
        learning_rate = step_iterator(beg = 0.0001, last = 0, step = 0)
        weight_decay = step_iterator(beg = 0, last = 0, step = 0)
        momentum = step_iterator(beg = 0.5, last = 0.9, step = 0.1)
        layers_dict = {0: {
                            'dimension':X.shape[1],     
                            'bias': None,
                            'value': None,
                            'layer_type': 'softmax',
                            'layer_name': 'input'
                         },
                      1: {
                            'dimension':50,     
                            'bias': None,
                            'value': None,
                            'layer_type': 'binary',
                            'layer_name': 'hidden'
                         }
              }
        batch_func = batch_func_generator(X, batch_size = 100)
        rbm = RSMRBM(layers_dict, weight_list = None, random_state = 20)
        rbm.fit(batch_func, PCD = False, error_function = 'recon',learning_rate = learning_rate, momentum = momentum,
                weight_decay = weight_decay, k = 1, perst_size = 100, n_iter = 1000, verbose = True)
    testRSMRBM()