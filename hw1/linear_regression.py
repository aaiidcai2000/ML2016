import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *


def linear_regression(X,Y):
    w,b=symbols('w b')
    residual = 0

    for i in range(10):
        residual += (Y[i]-w*X[i]-b)**2

    g1 = diff(residual,w)
    g2 = diff(residual,b)


    res=solve([g1,g2],[w,b])
    
    return res[w],res[b]



#data = pd.read_csv('train.csv',encoding="utf-8")
data = pd.read_csv('train.csv',encoding="latin1")
#data = pd.read_csv('train.csv',encoding="gb18030")

in_x = list(map(float,data.iloc[9:190:18,11].tolist()))
in_y = list(map(float,data.iloc[9:190:18,12].tolist()))

w,b=linear_regression(in_x,in_y)


residual = 0
for i in range(10):
    residual += (in_y[i]-w*in_x[i]-b)**2
print('redisual: ',residual,", w: ",w,", b: ",b)

#plot
LR_X = in_x
h= lambda x:b+w*x
F = np.vectorize(h)
LR_Y=F(LR_X)

plt.plot(LR_X,LR_Y)
plt.plot(in_x,in_y,'ro')
plt.show()
    



