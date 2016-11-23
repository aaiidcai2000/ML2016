import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *

def F(w,b,X):
    h= lambda x:b+w*x
    f = np.vectorize(h)
    return f(X)

def linear_regression(X,Y):
    w,b=symbols('w b')
    residual = 0

    for i in range(len(Y)):
        residual += (Y[i]-w*X[i]-b)**2

    g1 = diff(residual,w)
    g2 = diff(residual,b)
    print(g1)

    res=solve([g1,g2],[w,b])
    
    return res[w],res[b]

def plotResult(X,Y,w,b):
    LR_X = X
    LR_Y=F(w,b,LR_X)

    plt.plot(LR_X,LR_Y)
    plt.plot(X,Y,'ro')
    plt.show()
    

def train(data_csv):

    data = pd.read_csv(data_csv,encoding="latin1")

    in_x = list(map(float,data.iloc[9:4320:18,11].tolist()))
    in_y = list(map(float,data.iloc[9:4320:18,12].tolist()))

    w,b=linear_regression(in_x,in_y)

    residual = 0
    for i in range(len(in_y)):
        residual += (in_y[i]-w*in_x[i]-b)**2
    print('redisual: ',residual,", w: ",w,", b: ",b)

    #plotResult(in_x,in_y,w,b)
    
    return w,b

def test(data_csv,w,b):

    data = pd.read_csv(data_csv,header=None)

    x = list(map(float,data.iloc[9:4320:18,10].tolist()))

    y = F(w,b,x) 

    res = pd.DataFrame({
            'id':['id_'+str(i) for i in range(len(y))],
            'value':y,
        })

    #print(res)
    return res


w,b=train('train.csv')

#res = test('test_X.csv',w,b)

#res.to_csv('linear_regression_basic.csv',index=False )













