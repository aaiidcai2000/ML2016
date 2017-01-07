import sys
import pickle
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import Adam
from keras.utils import np_utils

threshold=0.85
nb_classes=10
batch_size = 50
nb_epoch=45

#handle data
f=open(sys.argv[1]+'/all_label.p','rb')
data = pickle.load(f)
data = np.array(data)

x_train = np.reshape(data,(-1,3,32,32)).astype('float32')
y_train = np.array(list( i for i in range(nb_classes) for j in range(500) ))


f = open(sys.argv[1]+'/all_unlabel.p','rb')
data = pickle.load(f)
data=np.array(data).reshape(-1,3,32,32).astype('float32')

model=load_model(sys.argv[2]+'_1')

res = model.predict(data)

addX=[]
addY=[]
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        if res[i][j]>threshold : 
            addX.append(i)
            addY.append(j)
            break

addY=np.array(addY)


x_train=np.append(x_train,list(data[addX[i]] for i in range(len(addX))),axis=0)
y_train=np.append(y_train,addY)

print(y_train.shape)
y_train = np_utils.to_categorical(y_train,nb_classes)

sample_weight=np.array(list( 1 if i < 500 else 0.5  for i in range(y_train.shape[0])))

#define model
model=Sequential()


model.add(Convolution2D(
    32,3,3,border_mode='same',dim_ordering='th',input_shape=x_train.shape[1:]
))
model.add(Activation('relu'))

model.add(Convolution2D(32,3,3,border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        shuffle=True,
        verbose=1,
        sample_weight=sample_weight,
        #validation_split=0.1,
)

model.save(sys.argv[2]+'_085')

#res = model.predict(x_train)
#loss,accuracy = model.evaluate(x_train,y_train)
#print("loss: ",loss," accuracy: ",accuracy)
