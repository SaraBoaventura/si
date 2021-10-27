# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:36:31 2021
@author: 35192
"""


import numpy as np
from copy import copy
from scipy import stats
import warnings 


class VarianceThreshold:
    
    def __init__(self,threshold=0):
        
        if threshold < 0 :
            warnings.warn("the threshod must be a non-negative value")
        self.threshold = threshold
        
    
    
    def fit (self, dataset):
        X = dataset.X
        self.var = np.var(X,axis=0)
        
    def transform(self, dataset, inline=False):
        X = data.set.X
        # for i in range (X.shape[1]):
        #     indx = []
        #     if self.var[i]>self.threshold:
        #         indx.append(i
        # outra forma 
        cond = self.var>self.threshold # cria um array de booleano 
        idxs = [i for i in range(len(cond)) if cond[i]]
        x_trans = X[:,idxs]
        xnames = [dataset._xnames[i] for i in idxs]
        if inline:
            dataset.X = X_trans 
            dataset._xnames = xnames 
            return dataset
        else:
            from.dataset import Dataset 
            return Dataset(copy(X_trans)), copy 