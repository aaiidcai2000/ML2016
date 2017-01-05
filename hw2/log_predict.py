import sys
import numpy as np
import pandas as pd


def F(X,theta):
    y=[]
    for i in range(len(X)):
        sum_up = theta[57]
        for j in range(57):
            sum_up+=theta[j]*X[i][j]
        y.append(0 if sum_up<0.5 else 1)
    return y
    
def test(data,theta):
    data=pd.read_csv(data,header=None)
    x=data.iloc[:,1:].apply(lambda x:pd.to_numeric(x)).as_matrix()
    
    
    y=F(x,theta)

    res=pd.DataFrame({
            'id':[i for i in range(1,len(y)+1)],
            'label':y,
        })
    return res

#w,b
with open(sys.argv[1],'r') as file:
    theta=file.read().splitlines()

theta=list(map(float,theta))

res=test(sys.argv[2],theta)

res.to_csv(sys.argv[3],index=False)

