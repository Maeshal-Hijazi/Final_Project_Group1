import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('movies.csv')
df = df[['Runtime (Minutes)', 'Metascore', 'Rating', ]]  # picking the needed columns
# and rearranging them to make the output column at the end.

df = df.fillna(df.mean().round(3))  # Filling missing cells with their column average

X = df.iloc[:, 0:-1].values  # All input features

x1 = df.iloc[:, 0].values  # Runtime (Minutes)
x2 = df.iloc[:, 1].values  # Metascore
y = df.iloc[:, -1].values  # Output: Rating

plt.scatter(x1, y)
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')
plt.show()

plt.scatter(x2, y)
plt.xlabel('Metascore')
plt.ylabel('Rating')
plt.show()

from statsmodels.stats import weightstats as stests

ztest, pval = stests.ztest(y, x1)
print(float(pval))
if pval < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

ztest, pval = stests.ztest(y, x2)
print(float(pval))
if pval < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

# Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(hidden_layer_sizes=(100, 100), solver='lbfgs', max_iter=5000)
regr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

pred = regr.predict(X_test)
print("The mean squared error:", mean_squared_error(y_test, pred))

z = np.arange(250)
e = y_test - pred

plt.plot(z, y_test)
plt.plot(z, pred)
plt.legend(['y_test', 'y_pred'])
plt.ylabel('y')
# plt.xlim(0,50)
plt.show()
plt.plot(z, e)
plt.ylabel('Error')
plt.show()

