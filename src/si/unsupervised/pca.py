
import numpy as np
from si.util.scale import StandardScaler

class PCA:
    '''
    A análise de componentes principais ou PCA, é um método de machine learn 
    não supervisionado que descobre conjuntos de variáveis correlacionadas 
    e as reduz para um conjunto de variáveis não correlacionadas que 
    representam as variáveis originais e são chamadas de componentes principais

    Os dados que mais explicam a variancia

    Objetivo: reduzir o número de dimensões de um conjunto de dados,
    podendo ser usado para melhor visualização, análise ou para compressão
    dos dados

    A matriz de covariância é uma matriz quadrada que contém as variâncias e covariâncias 
    associadas a diversas variáveis. Os elementos diagonais da matriz contêm os desvios das 
    variáveis, e os elementos fora da diagonal contêm as covariâncias entre todos os possíveis pares de variáveis.
    '''

    def __init__(self, ncomponents=2, using="svd"):
        # ncomponents must be int
        if ncomponents > 0 and isinstance(ncomponents, int):
            self.ncomponents = round(ncomponents)
        else:
            raise Exception("Number of components must be non negative and an integer")
        self.type = using
    

    def transform(self, dataset):
        scaled = StandardScaler().fit_transform(dataset).X.T   #normaliza os dados utilizando o standardscaler

        # numpy.linalg.svd:
        if self.type.lower()  == "svd": 
            self.u, self.s, self.vh = np.linalg.svd(scaled) # vetores e valores proprios
        else:
            self.cov_matrix = np.cov(scaled)                  #matriz de covariância dos dados normalizados
            # s are eigenvalues (valores proprios), u are eigenvectors(vetores proprios)
            self.s, self.u = np.linalg.eig(self.cov_matrix)   # Compute the eigenvalues and eigenvectors
        
        self.idx = np.argsort(self.s)[::-1]   #ordenação dos idx (ordem decrescente) por importancia de componentes
        self.eigen_val =  self.s[self.idx]    #reorganizaçao dos valores proprios pelos idx das colunas
        self.eigen_vect = self.u[:, self.idx] #reorganizaçao dos vetores proprios pelos idx das colunas

        self.sub_set_vect = self.eigen_vect[:, :self.ncomponents]  #vetores ordenados
        return scaled.T.dot(self.sub_set_vect) # produto dos dois arrays


    def variance_explained(self):  
        # variabilidade dos dados
        sum_ = np.sum(self.eigen_val)
        for i in self.eigen_val:
            percentage = i / sum_ * 100 
        return np.array(percentage)


    def fit_transform(self, dataset):
        '''
        corre o algoritmo sobre uma
        matriz de dados numéricos
        '''
        trans = self.transform(dataset)
        exp = self.variance_explained()
        return trans, exp