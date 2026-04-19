import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
# from sklearn.svm import SVC
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path('data/chest_xray')

def StandardScaler(X: np.ndarray):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / (std + 1e-9)
    
    return X_scaled

class SVM:
    def __init__(self, lr: float = 0.01, C: float = 10, n_epochs = 100):
        self.lr = lr
        self.C = C
        self.lam = 1/C
        self.n_epochs = n_epochs
        self.loss_his = []
    
    def hingeloss(self, hinge_sum):
        return 0.5 * (self.w.T @ self.w).item() + self.C * hinge_sum
    
    def fit(self, X, y):
        m, n = X.shape
        ones = np.ones((m, 1))
        X_bar = np.concatenate([X, ones], axis = 1)
        y = y.reshape(-1, 1)
        self.w = np.zeros((n+1, 1)) # Have bias

        # SGD
        for i in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffle = X_bar[indices]
            y_shuffle = y[indices]
            hinge_sum = 0
            for idx in range(m):
                xi = X_shuffle[idx: idx+1]
                yi = y_shuffle[idx: idx+1]
                y_hat = xi @ self.w 
                condition = y_hat * yi
                if condition < 1:
                    deri = -yi * xi.T
                else:
                    deri = 0
                delta = deri + self.lam * self.w
                self.w = self.w - self.lr * delta
                hinge_sum += max(0, 1 - condition)
            loss = self.hingeloss(hinge_sum)
            self.loss_his.append(loss)
        
    def predict(self, X):
        m, n = X.shape
        ones = np.ones((m, 1))
        X_bar = np.concatenate([X, ones], axis = 1)
        res = X_bar @ self.w
        res = np.where(res>=0, 1, -1).flatten()
        return res
    
def load_data(split: str = 'train'):
    normal = "NORMAL"
    pneunomia = "PNEUMONIA"

    images = []
    labels = []

    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):
        image = cv2.imread(os.path.join(BASE_DIR, split, normal, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_NEAREST)
        images.append(image)
        labels.append(1) # normal

    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneunomia)):
        image = cv2.imread(os.path.join(BASE_DIR, split, pneunomia, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_NEAREST)
        images.append(image)
        labels.append(-1) # normal

    return images, labels

train_images, train_labels = load_data("train")
test_images, test_labels = load_data("test")

train_images = [img.flatten() for img in train_images]
test_images = [img.flatten() for img in test_images]

train_images, train_labels = np.array(train_images), np.array(train_labels)
test_images, test_labels = np.array(test_images), np.array(test_labels)
# Scaling
train_images = StandardScaler(train_images)
test_images = StandardScaler(test_images)

cls = SVM(lr = 0.01, C= 10 , n_epochs=100)
cls.fit(train_images, train_labels)
pred = cls.predict(test_images)
print(f'F1: {f1_score(pred, test_labels):.4f}')
print(f"Precision: {precision_score(pred, test_labels):.4f}")
print(f"Recall: {recall_score(pred, test_labels):.4f}")
print(f"Loss history: {cls.loss_his}")


