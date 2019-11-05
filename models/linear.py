import numpy as np

class linear_regresser:
    def __init__(self):
        self.coeffs = None
        self.R_train = None
        self.sq_error = None
        self.n_features = None
        self.n_data = None
        
    def _add_bias_dim(self,X):
        new_column = np.ones((X.shape[0],1))
        return np.append(X,new_column,axis = 1)
    
    def _evaluate_errors(self,y,y_pred):
        sq_error = sum((y[i]-y_pred[i])**2 for i in range(len(y)))
        y_mean = np.mean(y)
        variance = sum((y[i]-y_mean)**2 for i in range(len(y)))
        return 1- sq_error/variance, sq_error
        
        
    def fit(self,X,y):
        self.n_data,self.n_features = X.shape
        Xa = self._add_bias_dim(X)
        self.coeffs = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ y
        y_pred = Xa @ self.coeffs
        self.R_train,self.sq_error = self._evaluate_errors(y,y_pred)
    
    def predict(self,X,return_errors = False):
        Xa = self._add_bias_dim(X)
        y_pred = Xa @ self.coeffs
        
        if return_errors: 
            R,sq_error = self._evaluate_errors(y,y_pred)
            return y_pred,R,sq_error
        return y_pred