# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:36:31 2021
@author: 35192
"""


import numpy as np
from copy import copy
from scipy import stats
import warnings 


from si.data import Dataset


class VarianceThreshold:

    
    def __init__(self,threshold=0):
        
        if threshold < 0 :
            warnings.warn("the threshod must be a non-negative value")
            threshold = 0
        self.threshold = threshold
        
    
    def fit(self, dataset):
        '''
        recebe o dataset e calcula a variancia
        '''
        X = dataset.X
        self._var = np.var(X,axis=0)
        


    def transform(self, dataset, inline=False):
        """metodo transform - recebe o dataset e elimina no dataset 
        todas as features que têm uma variância igual ou inferior ao treshold"""
        X = dataset.X
        cond = self._var > self.threshold #array de booleanos se a variancia for maior que o threshold e True
        idxs = [i for i in range(len(cond)) if cond[i]] # só ha o i do cond = True, assim so ha os indicies daqueles que a variancia é maior que o thershold
        X_trans = X[:,idxs] # todas as linhas e so as colunas a cima "selecionadas"
        xnames = [dataset.xnames[i] for i in idxs] # percorremos o array de boolenos  e vai se o xnames daqueles que a variancia é maior que o threshold
        if inline: # se for verdadeiro
            dataset.X = X_trans
            dataset.xnames = xnames
            return dataset
        else:
            return Dataset(copy(X_trans), copy(dataset.Y), xnames, copy(dataset.yname))
    

    def fit_transform(self, dataset, inline=False):
        '''
        recebe o dataset e transforma-o com a funçao transform
        '''
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


class SelectKBest:
    '''
    Escolhe as melhores features do nosso dataset
    As que tem o melhor score 
    dependendo das funçoes classificaçao ou regressao 

    Escolher as features que mais contribuem para a nossa variavel target, aquilo
    que queremos prever, que é o y. 
    '''

    def __init__(self, k: int, score_funcs):

        if score_funcs in (f_classification, f_regression):
            self._func = score_funcs

        if k > 0:
            self.k = k
        else:
            warnings.warn('Invalid feature number, K must be greater than 0.')
        
        
    def fit(self, dataset):
        self.F, self.P = self._func(dataset)

    def transform(self, dataset, inline=False):
        X = dataset.X
        xnames = dataset.xnames
        feat_selection = sorted(np.argsort(self.F)[-self.k:]) # -SELF.K VAI BUSCAR OS ultimos das lista pois o sorted faz por ordem crescente, ou seja os maiores valores vao estar no fim 
        x = X[:, feat_selection]
        xnames = [xnames[feat] for feat in feat_selection]

        if inline:
            dataset.X = x
            dataset.xnames = xnames
            return dataset
        else:
            return Dataset(x, copy(dataset.Y), xnames, copy(dataset.yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)





def f_classification(dataset):
    '''
    
    ''' 
    X = dataset.X
    Y = dataset.Y

    args = [X[Y == a, :] for a in np.unique(Y)]
    F, p = stats.f_oneway(*args)

    return F, p


def f_regression(dataset):
    """
    Grau de liberdade é, em estatística, o número de determinações independentes
    menos o número de parâmetros estatísticos a serem avaliados na população.
    """

    from scipy.stats import f
    
    X = dataset.X
    Y = dataset.Y

    correlation_coeficient = np.array([stats.pearsonr(X[:,i], Y)[0] for i in range(X.shape[1])])
    deg_of_freedom = Y.size - 2
    corr_coef_squared = correlation_coeficient ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = f.sf(F, 1, deg_of_freedom)

    return F, p