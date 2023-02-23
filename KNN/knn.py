import numpy as np


class KNNClassifier:
    
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):

        result = np.zeros(shape=(len(X), len(self.train_X)))
        for i in range(len(X)):
            for j in range(len(self.train_X)):
                result[i, j] = np.abs(self.train_X[j]-X[i]).sum()
        return result


    def compute_distances_one_loop(self, X):

        result = np.zeros(shape=(len(X), len(self.train_X)))
        for i in range(len(X)):
            result[i] = np.sum(np.abs(self.train_X-X[i]), axis=1)
        return result


    def compute_distances_no_loops(self, X):

        result = np.sum(np.abs(self.train_X-X[:,np.newaxis]), axis=2)
        return result

    
    def predict_labels_binary(self, distances):
    
        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, dtype=int)
        
        if self.k == 1:
            
            a = distances.argmin(axis=1)
            prediction = self.train_y[a]
            return prediction

        if self.k > 1:
            
            a = distances.argsort(axis=-1)[:,:self.k]
            mins = np.zeros((n_test, self.k), dtype=int)
            
            for i in range(self.k):
                new_arr = a[:,i]
                mins[:,i] = self.train_y[new_arr]

            for i in range(len(mins)):
                prediction[i] = np.bincount(mins[i]).argmax()

            return prediction.astype(object)
                        


    def predict_labels_multiclass(self, distances):
        
        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, dtype=int)
        
        if self.k == 1:
            
            a = distances.argmin(axis=1)
            prediction = self.train_y[a]
            return prediction
        
        if self.k > 1:
            
            a = distances.argsort(axis=-1)[:,:self.k]
            mins = np.zeros((n_test, self.k), dtype=int)
            
            for i in range(self.k):
                new_arr = a[:,i]
                mins[:,i] = self.train_y[new_arr]

            for i in range(len(mins)):
                prediction[i] = np.bincount(mins[i]).argmax()
            
            return prediction
