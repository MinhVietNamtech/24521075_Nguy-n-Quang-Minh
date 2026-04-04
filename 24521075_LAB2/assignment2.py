from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

class SoftmaxRegression_scratch():
    def __init__(self, lr, epochs, n_class):
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.losses=[]

    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # prevent overflow risk
        A = e_Z / e_Z.sum(axis = 1, keepdims = True)
        return A
    
    def ohe(self, y):
        print(type(y))
        m = np.array(y).ravel().shape[0]
        n = self.n_class
        labels = np.zeros((m,n))
        for i, val in enumerate(y):
            labels[i][val] = 1
        return labels
        
    def loss_function(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-25, 1 - 1e-15)
        cross_ent = np.sum(y * np.log(y_hat), axis=1)
        return float(-np.mean(cross_ent))
    
    def fit(self, X, y):
        # convert y to Y (one-hot encoding)
        Y = self.ohe(y)
        m, n = X.shape
        self.Theta = np.random.randn(n, self.n_class) * 0.1
        for e in range(self.epochs):
            y_hat = np.dot(X, self.Theta)
            A = self.softmax(y_hat)
            loss = self.loss_function(Y, A)
            self.losses.append(loss)
            E = A - Y
            gradient = (np.dot(X.T, E) + 0.1*self.Theta) / m
            self.Theta -= self.lr * gradient

    def predict(self, X):
        Y_pred = np.dot(X, self.Theta)
        pred = np.argmax(Y_pred, axis=1)
        return pred
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        recall = recall_score(y, y_hat, average='macro')
        precision = precision_score(y, y_hat, average='macro')
        f1 = f1_score(y, y_hat, average='macro')

        return {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }

def flatten(z):
    N,_,_ = z.shape
    new_data = z.reshape(N, -1)
    return new_data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Flatten image
x_train = flatten(x_train)
x_test = flatten(x_test)

model = SoftmaxRegression_scratch(lr=0.1, epochs=300, n_class=10)
model.fit(x_train, y_train)
metrics = model.evaluate(x_test, y_test)
for met, score in metrics.items():
    print(f'{met}: {score}')
y = model.losses
x = range(1, len(y)+1)
plt.figure(figsize=(10,10))
plt.plot(x, y, linestyle='-', linewidth = 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss function of training process')
plt.show()