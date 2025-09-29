import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model
data = pd.read_csv('data.csv',header=None)
x=data[1434]
y=data[1435]
print(x,y)
features=data[[i for i in range(0,1434)]]
x=x.to_numpy()
y=y.to_numpy()
features=features.to_numpy()
print(y)
clf_x=linear_model.SGDRegressor()

clf_x.fit(features,x)
print('donex \n\n\n\n\n')

print('doney \n\n\n\n\n')
pickle.dump(clf_x,open('x_solomon.pkl','wb'))
clf_y=linear_model.SGDRegressor()
clf_y.fit(features,y)
pickle.dump(clf_y,open('y_solomon.pkl','wb'))