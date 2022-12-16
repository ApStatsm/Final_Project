# OpenSource SW 2022-2 Final_Project 
--------------------------------------
### Name: Sangmin Lee
### ID: 20212344
### Department: Applied Statistics
### Double major: Artificial Intelligence
--------------------------------------
# Brain Tumor Classfication Using Scikit-learn model

## Description

In this project, we used Python's scikit-learn package to classify the types of brain tumors. </br>
I tried many methods and hyperparameter combinations in order to search appropirate model for classifying. </br>
These are the steps to reach conclusion.

## Project info
### 1. Dataset
The types of brain tumors were classified into the following four categories. </br>
- glioma-tumor
- meningioma-tumor
- pituitary-tumor
- no-tumor
</br>
and in each category folders, mri images are included. 
these datasets were given by professor.

### 2. Import packages
```python
import os

import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics

import skimage.io
import skimage.transform
import skimage.color

import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline
```
Import packages that are needed to execute the codes. </br>
- import os to load dataset.
- import sklearn for machine learning.
- import skimage for load image data.
- import numpy to reshape the image data.
- import matplotlib to print the image.

```python
import sys, os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
```
those are packages that i imported for my convenience. </br>

### 3. Load images
I need to load brain tumor images of each category to make classification model. </br>

```python
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
```
Those codes load image data and reshape them as array in X, y. </br>
From now on, I can use X, y datas in analysis. </br>

```python
j = 0
for i in range(len(y)):
    if y[i] in labels[j]:
        plt.imshow(images[i])
        plt.title("[Index:{}] Label:{}".format(i, y[i]))
        plt.show()
        j += 1
    if j >= len(labels):
        break
```
print first images from each category. </br>
I can check whether images are imported properly. </br>

### 4. Classification with Scikit-learn models.

```python
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
```
Split data as train and test data in ratio of 7:3. </br>
In this way, I can train the model with train data and test the model with test data. </br>

```python
clf = sklearn.< model >( < hyperparameters > )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```
I defined clf as the classification model with tuned hyperparameter. </br>
and fit the train data in clf. </br>
Then predict the label of test data point(assign in y_pred). </br>

```python
print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```
as a result, test the trained model with test data and print the model's accuracy.

### 5. How to search appropirate model
Like all the machine learning models, the accuracy of the model is very important.
Then, how do I can make it more accuarate?
Most of all, searching good classification module and tuning hyperparameter is really importatnt.

First, I needed to search as many as possible classification modules in scikit-learn package.
In scikit-learn, there are many classifiers.
- Logistic Regression
- Gradient Boosting Classifier
- Random Forest Classifier
- Perceptron
- K Neighbors Classifier
... etc

I checked default accuracy of each modules to use high accuarcy modules. </br>
And I decided to tune the hyperparameters of those modules. </br>
- Multi-Layer Perceptron Classfier
- Perceptron
- Support Vector Machine Classifier
- K Neighbors Classifier

There are many way to search best hyperparameters, but I used most common way 'GridSearchCV'. </br>
In GridSearchCV, we can search best hyperparameters by applying all hyperparameter variables. </br>

For example, I will apply GridSearchCV in Support Vector Machine Classifier.
```python
# module assignmnet
svm = SVC()

# create a grid of hyperparameters to test
param_svm = {'C': [0.1,1, 10, 100],
'gamma': [1,0.9,0.1,0.01,0.001],
'kernel': ['rbf', 'poly', 'sigmoid']
}
# test hyperparameter
gscv_svm = GridSearchCV (estimator = svm, param_grid = param_svm, scoring ='accuracy', cv = 3, refit=True, n_jobs=1, verbose=2)

# fit the trained data
gscv_svm.fit(X_train, y_train)

# print accuracy of best hyperparameter
print('svm Parameter: ', gscv_svm.best_params_)
print('svm estimate accuracy: {:.4f}'.format(gscv_svm.best_score_))
```
In GridSearch variable, we can designate scoring method such as r2, ROC etc..
cv means number of division to test.
I designate scoring method as 'accuracy' and cv as '3'.

in this way we can check what combination of hyperparameters are best.

And in the Project IPYNP file, I wrote the best module and hyperparameter I tested.

### 6. About hyperparameter of my model
There are 3 parameters that I tuned in Support Vector Machine.
- C
- gamma
- kernel

First, C means the penalty of error term.
Large C value means thatit gives greater penalties to the misclassfication of the model.
If it is too large, it has risk of overfitting, and if it is too small, it has risk of underfitting.

Second, gamma means the degree that  adjacent values are classfied into the same label.
It also determines the curve rate of the decision boundaries.

Last, kernel specifies the kernel type to be used in the algorithm.
To put it simply it is kind of dimensional conversion to classify the models.



