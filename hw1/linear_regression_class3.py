import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *

def F(w,b,X):
    y=[]
    for j in range(X.shape[0]):
        temp=b
        for i in range(len(X[0])):
            temp+= w[i]*X[j,i]
        y.append(temp)
    return y


def linear_regression(X,Y,rate):
    W=symbols('w0:153')
    b=symbols('b')
    w0=np.zeros(153)
    b0=0
    lastR = 2**31-1
    R=lastR-1
    
    threshold = 4000

    Loss=0
    for i in range(len(Y)):
        guess=b
        for j in range(len(X[0])):
            guess +=  X[i,j]*W[j]
        Loss+= (Y[i]-guess)**2
    print("Loss: ",expand(Loss))

    residual = lambdify((W,b),Loss,"numpy")

    
    G1 = list(diff(Loss,W[i]) for i in range(len(W)))
    G2 = diff(Loss,b) 

    for i in range(len(W)):
        G1[i]= lambdify((W,b),G1[i],"numpy") 
    G2= lambdify((W,b),G2,"numpy")
    

    #for j in range(2):   
    while(R<= lastR):
        lastR=R
        R = residual(w0,b0)
        print('redisual: ',R)
        if(R<threshold):
            threshold-=250
            rate*=1.02
        '''
        print("W: ",end="")
        for i in range(len(W)):
            print(w0[i],end="," )
        print(" B: ",b0)
        '''
        for i in range(len(W)):
            w0[i] -= rate*G1[i](w0,b0)
        b0 -= rate*G2(w0,b0)
        

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
    data.drop(data.index[10:4320:18],inplace=True)
    in_x = data.iloc[:,3:12].apply(lambda x:pd.to_numeric(x)).as_matrix().reshape(240,153)
    in_y = list(map(float,data.iloc[9:4080:17,12].tolist()))
    
    w,b=linear_regression(in_x,in_y,learningRate)

    #plotResult(in_x,in_y,w,b)
    
    return w,b

def test(data_csv,w,b):

    data = pd.read_csv(data_csv,header=None)
    data.drop(data.index[10:4320:18],inplace=True)

    x = data.iloc[:,2:11].apply(lambda x:pd.to_numeric(x)).as_matrix().reshape(240,153)
    #x = list(map(float,data.iloc[9:4320:18,10].tolist()))
    
    y = F(w,b,x) 
    
    res = pd.DataFrame({
            'id':['id_'+str(i) for i in range(len(y))],
            'value':y,
        })

    #print(res)
    return res

w,b=train('train.csv',0.000000187)

res = test('test_X.csv',w,b)

res.to_csv('linear_regression_class3.csv',index=False )




