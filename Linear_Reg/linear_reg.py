# a simple and basic version of linear regression.

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearRegression:

    def __init__(self,lr=0.002,n_iter=1000):
        self.lr = lr
        self.n_iter= n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias=0

        y_pred = np.dot(X,self.weights)+self.bias

        #calculate error and update params using gradient descent
        dw = (1/n_samples)* np.dot(X.T,(y_pred-y))
        db =  (1/n_samples) * np.sum(y_pred-y)

        self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db

    def predict(self,X):

        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred


    def mse(self,y_test,predictions):
        return np.mean((y_test-predictions)**2)


X, y = datasets.make_regression(n_samples=100,n_features=1,noise=15,random_state=69)
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

reg = LinearRegression(lr=0.05) #change the learning rates and experiment
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

MSE = reg.mse(y_test,predictions)
print("The Mean square error:",MSE)