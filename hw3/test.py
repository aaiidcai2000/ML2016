import sys
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model

f = open(sys.argv[1]+'/test.p','rb')
data = pickle.load(f)
data=np.array(data['data']).reshape(-1,3,32,32).astype('float32')

model=load_model(sys.argv[2])

res = model.predict(data)

predictY=[]
for i in range(res.shape[0]):
    max=res[i][0]
    ele_class=0
    for j in range(1,res.shape[1]):
        if res[i][j]>max :
            max=res[i][j]
            ele_class=j
    predictY.append(ele_class)
            
#print(predictY)
print(len(predictY))

res = pd.DataFrame({
    'ID':[i for i in range(len(predictY))],
    'class':predictY,

})
res.to_csv(sys.argv[3],index=False)

