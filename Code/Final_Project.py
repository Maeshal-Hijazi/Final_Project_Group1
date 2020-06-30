import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('movies.csv')
df = df[['Runtime (Minutes)','Votes','Metascore','Rating']] # picking the needed columns
# and rearranging them to make the output column at the end.

df = df.fillna(df.mean().round(3))  # Filling missing cells with their column average


X = df.iloc[:, 0:3].values  # All input features

x1 = df.iloc[:, 0].values  # Runtime (Minutes)
x2 = df.iloc[:, 1].values  # Vote
x3= df.iloc[:, 2].values  # Metascore
y = df.iloc[:, -1].values  # Output: Rating

plt.scatter(x1, y)
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')
plt.show()

plt.scatter(x2, y)
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.show()

plt.scatter(x3, y)
plt.xlabel('Votes')
plt.ylabel('Metascore')
plt.show()

from scipy import stats
from statsmodels.stats import weightstats as stests

ztest, pval = stests.ztest(y, x1)
print('P-value for Runtime and Rating:', float(pval))
if pval < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

ztest, pval = stests.ztest(y, x2)
print('P-value for Votes and Rating:', float(pval))
if pval < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

ztest, pval = stests.ztest(y, x3)
print('P-value for Metascore and Rating:', float(pval))
if pval < 0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

# Splitting the data for training and testing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train2 = sc.transform(X_train)
X_test2 = sc.transform(X_test)


regr = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000)
regr.fit(X_train2, y_train)

from sklearn.metrics import r2_score, mean_squared_error

pred = regr.predict(X_test2)
print('----------Mean squared error---------- \n', mean_squared_error(y_test, pred))
# print(r2_score(y_test, pred))
z = np.arange(250)
e = y_test - pred

plt.plot(z, y_test, label='y_test')
plt.plot(z, pred, label='y_pred')
plt.legend(framealpha=1, frameon=True);
plt.xlabel('Sample')
plt.ylabel('Output Value')

plt.show()
plt.plot(z, e)
plt.xlabel('Sample')
plt.ylabel('Error')
plt.show()