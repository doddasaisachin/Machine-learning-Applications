import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:\\Users\\online\\Desktop\\codes\\Datasets\\iris.csv")
X = dataset.iloc[:, [1,2,3,4]].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
from sklearn import metrics
print("Classification Accuracy:", round(metrics.accuracy_score(y_test, y_pred)*100),"%")
import seaborn as sn
from matplotlib import pyplot as plt
plt.figure(figsize=(5,4))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
