import numpy as np
import pandas as pd
import math
import sys

def F(X,w,b):
    y=[]
    for i in range(len(X)):
        sum_up = b
        for j in range(57):
            sum_up+=w[j]*X[i][j]
        y.append(0 if sum_up<0.5 else 1)
    return y
    
def test_oriData(X,Y,w,b):
    resY=F(X,w,b)
    hit=0
    for i in range(len(Y)):
        if resY[i]==Y[i]: 
            hit+=1
    print("hit rate: ",hit/len(Y)," ",hit,"/",len(Y))

def sigmoid(z):
    return 1/ (1+np.exp(-z))

def update(X,Y,w,b,rate):
    totalE=0
    for i in range(len(Y)):
        e=sigmoid(w.T.dot(X[i])+b)-Y[i]
        totalE+=abs(e)
        
        for j in range(len(w)):
            w[j]-=rate*e*X[i][j]
        b-=rate*e
    
    return w,b,totalE

def logistic_regression(X,Y,rate):
    w=np.zeros(57)
    b=0
    lastE=21475668
    E=21475667
    for i in range(392):  #392
    #while(lastE<E):
        lastE=E
        w,b,E=update(X,Y,w,b,rate)
        rate*=0.98
        #print(i," ",E," ",rate)

    return w,b


def train(data_csv,learningRate):

    data = pd.read_csv(data_csv,header=None )
    X = data.iloc[:,1:58].apply(lambda x:pd.to_numeric(x)).as_matrix()
    Y = list(map(float,data.iloc[:,58]))

    
    w,b=logistic_regression(X,Y,learningRate)

    #test_oriData(X,Y,w,b)

    return w,b





w,b=train(sys.argv[1],0.01)

'''
with open(sys.argv[2],'w') as file:
    file.write("w = " + w )
    file.write("b = " + b )
'''

for i in range(len(w)):
    print(w[i])
print(b)



