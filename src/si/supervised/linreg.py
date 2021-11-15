import numpy as np
from si.util.metrics import mse 


class LinearRegression():

    def fit(self, dataset):
        x,y = dataset.getXy()
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.X = x
        self.Y = Y 
        #
        self.train_gd(x,y) if self.gd else self.train_closed(x,y)
        self.is_fitted = True
    
    def train_closed(self,x,y):
        """ Uses closed form linear algebra to fit the model.
        theta = inv(XT*X)*XT*Y
        """
        self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y))

    def train_gd(self,x,y):
        m = x.shape[0]
        n = x.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            grad = 1/m * (x.dot(self.theta)-y).dot(x)
            self.theta = self.lr * grad
            self.history[epoch] = [self.theta][:], self.cost()

    def predict(self,x):
        assert self.is_fitted,
        _x = np.hstack(([1],x))
        return np.dot(self.theta,_x)


    def cost(self):
        y_pred = np.dot(self.x,self.theta) # calcula as previsões
        return mse(self.y,y_pred)/2


###
class LinearRegressionReg(LinearRegression):

    def __init__(self,gd=False,epochs=1000,lr=0.001,lbd=1):
        super(LinearRegressionReg,self).__init__(gd=gd,epochs=epochs,lr=lr,ldb=lbd)
        self.lbd = lbd 

    def train_closed(self,x,y):
        """ Uses closed form linear algebra to fit the model.
        theta = inv(XT*X+lbd*I)*XT*Y
        """
        n =x.shape[1]
        identity = np.eye(n)
        identity[0,0] = 0
        self.theta = np.linalg.inv(x.T.dot(x)+self.lbd*identity).dot(x.T).dot(y)
        self.is_fitted = true
    
    def train_gd(self,x,y):
        m = x.shape[0]
        n = x.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m,self.lbd)
        lbds[0]=0
        for epoch in range(self.epochs):
            grad = 1/m * (x.dot(self.theta)-y).dot(x)
            self.theta = (self.lr/m) * (lbds+grad)
            self.history[epoch] = [self.theta[:], self.cost()]

   # def cost(self):
   #     h = sigmoid(np.dot)