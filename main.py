# working from here:
# https://thedatafrog.com/en/articles/handwritten-digit-recognition-scikit-learn/

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as numpy

digits = datasets.load_digits()
# look at the shape of the dataset
print(digits.images.shape)
# print out one images
i = 1001
print(digits.images[i])
print("###################")
plt.imshow(digits.images[i], cmap='binary')
plt.show()
plt.savefig('number.png')

#begin Lesson 3
#example fo supervised learning,
# we know a large amount of i/o
print(digits.target.shape)
print(digits.target[i])


## plot a series of digits
def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j + 1)
        plt.imshow(digits.images[i + j], cmap='binary')
        plt.title(digits.target[i + j])
        plt.axis('off')
    plt.show()
    plt.savefig('multiDigits.png')


plot_multi(0)

## building the network
y = digits.target  #output
x = digits.images.reshape(
    (len(digits.images),
     -1))  #create a 1-D (flattened) array of all the 2-D arrays)
print(x.shape)
print(x[i])

##split into training and sampling datasets
x_train = x[:1000]  #1st 1000 images
y_train = y[:1000]  #0-999
x_test = x[1000:]  #rest of the data
y_test = y[1000:]  #1000-1796

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(15, ),
    activation='logistic',
    alpha=1e-4,
    solver='sgd',
    tol=1e-4,
    random_state=1,
    learning_rate_init=.1,
    verbose=True)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
print("Prediction:  ")
print(predictions[:50])  #look at 1st 50 (0-49) from x_test list
print(y_test[:50])

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
