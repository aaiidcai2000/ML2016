import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sympy import *

def F(w,b,X):
    h= lambda x:b+w*x
    f = np.vectorize(h)
    return f(X)


def linear_regression(X,Y,rate):
    w,b=symbols('w b')
    w0=1
    b0=2
    lastR = 2**31-1
    R=lastR-1

    Loss=0
    for i in range(len(Y)):
        Loss += (Y[i]-w*X[i]-b)**2
    print("Loss: ",expand(Loss))
    residual = lambdify((w,b),Loss,"numpy")
    g1 = diff(Loss,w)
    g2 = diff(Loss,b)
    print("g1: ",expand(g1))
    print("g2: ",expand(g2))
    g2=lambdify((w,b),g2,"numpy")
    g1=lambdify((w,b),g1,"numpy")

    while(R<= lastR):
        lastR=R
        R = residual(w0,b0)
        print('redisual: ',R,", w: ",w0,", b: ",b0)
        
        w0,b0=w0-rate*g1(w0,b0),b0-rate*g2(w0,b0)
        print("w1: ",w0)
        print("b1: ",b0)

    R = residual(w0,b0)
    print('redisual: ',R,", w: ",w0,", b: ",b0)


    return w0,b0

def plotResult(X,Y,w,b):
    LR_X = X
    LR_Y=F(w,b,LR_X)

    plt.plot(LR_X,LR_Y)
    plt.plot(X,Y,'ro')
    plt.show()
    

def train(data_csv,learningRate):

    data = pd.read_csv(data_csv,encoding="latin1")

    in_x = list(map(float,data.iloc[9:4320:18,11].tolist()))
    in_y = list(map(float,data.iloc[9:4320:18,12].tolist()))

    w,b=linear_regression(in_x,in_y,learningRate)

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


w,b=train('train.csv',0.000008)

#res = test('test_X.csv',w,b)

#res.to_csv('linear_regression_basic.csv',index=False )





