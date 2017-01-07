import pickle
import numpy as np
from keras.models import load_model
from keras.utils import np_utils


#for test ori accuracy
nb_classes=10

#handle data
f=open('data/all_label.p','rb')
data = pickle.load(f)
data = np.array(data)

x_train = np.reshape(data,(-1,3,32,32)).astype('float32')
y_train = np.array(list( i for i in range(nb_classes) for j in range(500) ))

y_train = np_utils.to_categorical(y_train,nb_classes)

model=load_model('model2')

loss,accuracy = model.evaluate(x_train,y_train)
print("loss: ",loss," accuracy: ",accuracy)
