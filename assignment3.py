from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

def filter_data(data, condition):
  images, labels = data

  new_images = images[labels == condition]
  new_labels = labels[labels == condition]

  return new_images, new_labels

def flatten(z):
  N,_,_ = z.shape
  new_data = z.reshape(N, -1)
  return new_data

def evaluate(y, y_hat):
        recall = recall_score(y, y_hat, average='macro')
        precision = precision_score(y, y_hat, average='macro')
        f1 = f1_score(y, y_hat, average='macro')
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }

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
model = LogisticRegression(max_iter=300)
model.fit(train_data, new_train_label)
# Metrics
y_pred = model.predict(test_data)
metrics = evaluate(y_pred, new_test_label)
print('Logistics Regression:')
for met, score in metrics.items():
    print(f'{met}: {score}')

# Softmax
soft_train_set = flatten(x_train)
soft_test_set = flatten(x_test)

model = LogisticRegression(max_iter=300, multi_class='multinomial')
model.fit(soft_train_set, y_train)
# Metrics
y_pred = model.predict(soft_test_set)
soft_metrics = evaluate(model.predict(soft_test_set), y_test)
print('Softmax Regression:')
for met, score in soft_metrics.items():
    print(f'{met}: {score}')


