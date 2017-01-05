import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sympy import *

def F(w,b,X):
    y=[]
    for j in range(X.shape[0]):
        temp=0
        for i in range(len(X[0])):
            temp+= b[i]+w[i]*X[j,i]
        y.append(temp)
    return y


def linear_regression(X,Y,rate):
    W=symbols('w0:9')
    B=symbols('b0:9')
    w0=np.zeros(9)
    b0=np.zeros(9)
    lastR = 2**31-1
    R=lastR-1
    

    Loss=0
    for i in range(len(Y)):
        guess=0
        for j in range(len(X[0])):
            guess +=  X[i,j]*W[j]+B[j]
        Loss+= (Y[i]-guess)**2
    print("Loss: ",expand(Loss))

    residual = lambdify((W,B),Loss,"numpy")

    
    G1 = list(diff(Loss,W[i]) for i in range(len(W)))
    G2 = list(diff(Loss,B[i]) for i in range(len(W)))

    for i in range(len(W)):
        G1[i]= lambdify((W,B),G1[i],"numpy") 
        G2[i]= lambdify((W,B),G2[i],"numpy")
    

    while(R<= lastR):
        lastR=R
        R = residual(w0,b0)
        print('redisual: ',R)
        
        print("W: ",end="")
        for i in range(len(W)):
            print(w0[i],end="," )
        print(" B: ",end="")
        for i in range(len(W)):
            print(b0[i],end="," )
        print("")
        
        for i in range(len(W)):
            w0[i] -= rate*G1[i](w0,b0)
            b0[i] -= rate*G2[i](w0,b0)
        
        print("W: ",end="")
        for i in range(len(W)):
            print(w0[i],end="," )
        print(" B: ",end="")
        for i in range(len(W)):
            print(b0[i],end="," )
        print("")
        

    R = residual(w0,b0)
    print('redisual: ',R)

    return w0,b0

def plotResult(X,Y,w,b):
    LR_X = X
    LR_Y=F(w,b,LR_X)

    plt.plot(LR_X,LR_Y)
    plt.plot(X,Y,'ro')
    plt.show()
    

def train(data_csv,learningRate):

    data = pd.read_csv(data_csv, encoding="latin1")
    #data.drop(data.index[10:4320:18],inplace=True)

    in_x = data.iloc[9:4320:18,3:12].apply(lambda x:pd.to_numeric(x)).as_matrix()
    in_y = list(map(float,data.iloc[9:4320:18,12].tolist()))
    
    w,b=linear_regression(in_x,in_y,learningRate)

    #plotResult(in_x,in_y,w,b)
    
    return w,b

def test(data_csv,w,b):

    data = pd.read_csv(data_csv,header=None)

    x = data.iloc[9:4320:18,2:11].apply(lambda x:pd.to_numeric(x)).as_matrix()
    #x = list(map(float,data.iloc[9:4320:18,10].tolist()))
    
    y = F(w,b,x) 
    
    res = pd.DataFrame({
            'id':['id_'+str(i) for i in range(len(y))],
            'value':y,
        })

    #print(res)
    return res

w,b=train('train.csv',0.0000001)

res = test('test_X.csv',w,b)

res.to_csv('linear_regression_class2.csv',index=False )




