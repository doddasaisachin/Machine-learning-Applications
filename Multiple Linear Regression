import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
dataset = pd.read_csv("C:\\Users\\online\\Desktop\\codes\\Datasets\\Advertising.csv")
print(dataset.head())
x = dataset[['TV', 'Radio', 'Newspaper']]
y = dataset['Sales']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()  
reg.fit(x_train, y_train)

print('Constant Factor: {}'.format(reg.score(x_test, y_test)))
print('Coefficients:')
print(list(zip(x, reg.coef_)))

y_pred=reg.predict(x_test)
y_pred

# Plotting the graph for the Testing dataset
  
plt.scatter(y_test,y_pred,color='blue');
plt.xlabel('Actual');
plt.ylabel('Predicted');
plt.show()
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');
plt.show()
