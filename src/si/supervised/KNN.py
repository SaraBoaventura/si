from si.supervised.model import Model
from si.util.util import l2_distance
from si.util.metrics import accuracy_score

class KNN(Model):

    def __init__(self,num_neighbors,classification = True):
        #super(KNN, self).__init__(num_neighbors,classification = True)    # instanciar a flag para saber se foi feito o fit ao modelo ou não
        self.num_neighbors = num_neighbors # K numero e vizinhos considerados no calculo do vizinho mais próximo
        self.classification = classification

    def fit(self,dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self,x):
        distances = l2_distance(x,self.dataset.X) # atraves da distacia euclidiana de acordo com o numero de vizinhos 
        sorted_index = np.argsort(distances) # ordenado as distancias, pois queremos adistancia mais proxima
        return sorted_index[:self.num_neighbors]

    def predict(self,x):
        assert self.is_fitted, 'Model must be fit before prediction'
        neighbors = self.get_neighbors(x)
        values = self.dataset.Y[neighbors].tolist()
        if self.classification:
            prediction = max(set(values), key = values.count) # aquele que aparece mais vezes vai ser a previsão
        else:                                       # problema de regressão
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis = 0, arr=self.datatset.X.T)
        return accuracy_score(self.dataset.Y,y_pred)


