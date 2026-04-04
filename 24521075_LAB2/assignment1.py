#ASSIGNMENT 1
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class LogisticRegression_scratch:
  def __init__(self, lr, epochs) -> None:
    self.epochs = epochs
    self.lr = lr
    self.losses = []

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z)) 

  def loss_fn(self, y, y_hat):
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    l = (1-y)* np.log(1-y_hat) + y*np.log(y_hat)
    return float(-np.mean(l))

  def fit(self, X, y):
    N, d = X.shape
    y = y.reshape(y.shape[0],-1)
    self.w = np.zeros((d, 1))
    for e in range(self.epochs):
      y_hat = self.sigmoid(np.dot(X, self.w))
      l = self.loss_fn(y, y_hat)
      self.losses.append(l)
      delta_y = (y_hat - y)
      gradient = np.dot(X.T, delta_y)
      self.w = self.w - self.lr * gradient

  def predict(self, X):
    return np.where(self.sigmoid(np.dot(X, self.w)) >=0.5, 1, 0)

  def evaluate(self, X_test, y_test):
    y_hat = self.predict(X_test)
    precision = precision_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
  
def filter_data(data, condition):
  images, labels = data

  new_images = images[labels == condition]
  new_labels = labels[labels == condition]

  return new_images, new_labels

def flatten(z):
  N,_,_ = z.shape
  new_data = z.reshape(N, -1)
  return new_data

# Apply in MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 0
train_ = tuple([x_train, y_train])
test_ = tuple([x_test, y_test])

train_data_0, train_label_0 = filter_data(train_, 0)
train_data_1, train_label_1 = filter_data(train_, 1)
test_data_0, test_label_0 = filter_data(test_, 0)
test_data_1, test_label_1 = filter_data(test_, 1)

new_train_image = np.concatenate([train_data_0, train_data_1], axis = 0)
new_train_label = np.concatenate([train_label_0, train_label_1], axis = 0)
new_test_image = np.concatenate([test_data_0, test_data_1], axis = 0)
new_test_label = np.concatenate([test_label_0, test_label_1], axis = 0)

# flatten image
train_data = flatten(new_train_image)
test_data = flatten(new_test_image)
# Do prediction
model = LogisticRegression_scratch(lr = 0.1, epochs=300)
model.fit(train_data, new_train_label)
metrics = model.evaluate(test_data, new_test_label)
for met, val in metrics.items():
  print(f'{met}: {val}')

y = model.losses
x = range(1, len(y)+1)

plt.figure(figsize=(10,10))
plt.plot(x, y, linestyle='-', linewidth = 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss function of training process')
plt.show()

