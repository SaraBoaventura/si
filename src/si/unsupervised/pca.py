
import numpy as np
from si.util.util import StandardScaler

class PCA:
     '''
    A análise de componentes principais ou PCA, é um método de machine learn 
    não supervisionado que descobre conjuntos de variáveis correlacionadas 
    e as reduz para um conjunto de variáveis não correlacionadas que 
    representam as variáveis originais e são chamadas de componentes principais

    Os dados que mais explicam a variancia
    '''

    __init__(self, ncomponents = 2, using = "svd"):
        # ncomponents must be int
        if ncomponents > 0 and isinstance(ncomponents, int):
            self.ncomponents = round(ncomponents)
        else:
            raise Exception("Number of components must be non negative and an integer")

    def fit(self):
        pass
    
    def transform(self, dataset):
        #Scale the features
        scaled = StandardScaler().fit_transform(dataset).X.T
    