import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('USD_TRY Historical Data.csv')['Price'].astype(float)
data = data[::-1]         # not getting the last one
data.index = pd.RangeIndex(len(data.index))

x = data.index
y = data
x = x.values.reshape(len(data),1)
y= y.values.reshape(len(data),1)

polynomial = PolynomialFeatures(degree=9)
Xnew = polynomial.fit_transform(x)
model = LinearRegression()
model.fit(Xnew,y)
P = model.predict(Xnew)
Q = []
for p in P:
    Q.append(p[0])
df1 = pd.DataFrame(Q)
df2 = pd.concat([data,df1], axis = 1)
df2.to_csv('learned.csv', sep=',')
Error = []
for d in range(len(y)):
    Error.append(abs(P[d]-y[d]))
plt.plot(x,y)
plt.plot(x,P)
plt.plot(x,Error, color= 'red')
plt.title('With  %.2f mean error in the last 50 days'%np.mean(Error[:50]))
plt.legend(['Real','Prediction','Error'])
plt.savefig('Prediction_of_Dollars_Rate.png',dpi = 100)
