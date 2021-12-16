from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import csv
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# X, y = load_iris(return_X_y=True)
data = pd.read_csv('drug200.csv')
sex  = {'M': 1, 'F': 0}
bp = {'HIGH': 2, 'LOW': 0, 'NORMAL': 1}
cholesterol = {'HIGH': 1, 'NORMAL': 0}
drug = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4}

data['BP'] = [bp[item] for item in data['BP']]
data['Cholesterol'] = [cholesterol[item] for item in data['Cholesterol']]
data['Drug'] = [drug[item] for item in data['Drug']]
data['Sex'] = [sex[gender] for gender in data['Sex']]


X = data.drop('Drug', axis=1)
y = data['Drug']

print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

results = []
gnb = GaussianNB()
nc = NearestCentroid()
rt = tree.DecisionTreeRegressor()

rtPrediction = rt.fit(X_train, y_train).predict(X_test)

ncPrediction = nc.fit(X_train, y_train).predict(X_test)
# for training and and testing of a function
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))
results.append(((y_test != y_pred).sum() / X_test.shape[0]) * 100)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != ncPrediction).sum()))
results.append(((y_test != ncPrediction).sum() / X_test.shape[0]) * 100)

print("Number of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != rtPrediction).sum()))
results.append(((y_test != rtPrediction).sum() / X_test.shape[0]) * 100)

print(X.shape, len(y))
print(X_train.shape, y_train.shape, X_test.shape)

plot_confusion_matrix(gnb, X_test, y_test)
plt.show()
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(['GaussianNB', 'NearestCentroid', 'regression'], results)
