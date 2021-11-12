from si.supervised import Model
from si.util.util import l2_distance, accuracy_score
from si.util.metrics import metrics

class KNN(Model):

    def__init__(self,num_neighbors):
        super(KNN).__init__()
        self.num_neighbors = num_neighbors


    def fit(self,dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self,x):
        distances = l2_distance(x,self.dataset.X)
        sorted_index = np.argsort(distances)
        return sorted_index[:self.num_neighbors]

    def predict(self,x):
        assert self.is_fitted, 'Model must be fit before prediction'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        prediction = max(set(values),key=values.count) # aquele que aparece mais vezes 
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis = 0, arr=self.datatset.X.T)
        return accuracy_score(self.dataset.y,y_pred)


