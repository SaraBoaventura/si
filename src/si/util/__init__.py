#from .util import *
#from .scale import *
#import numpy as np 
#import intertools

#class CrossValidationScore:

#    def __init__(self,model,dataset,**kwargs):
#        self.model = model
#        self.dataset = dataset
#        self.cv = kwargs.get('cv', 3)
#        self.split = Kwargs.get('split',0.8)
#        self.train_scores = None
#        self.test_scores = None
#        self.ds = None 

#    def predict (self,x):
#        assert self.is_filtted, 'Model must be fit before predicting'
#        _x = np.hstack(([1],x))
#        return np.dot(self.theta,_x)

#    def cost(self,X=None, y = None, theta=None):
#        X = add_intersect(X) if X is not None else self.x
#        y = y if y is not None else self.y 
#        theta = tehta if theta is not None else self.theta
#        y_pred = np.dot(X,theta)
#        return mse(y,y_pred)/2

# NAO TEM DE ESTAR NADA AQUI, VER ONDE SE COLOCA ISTO 

