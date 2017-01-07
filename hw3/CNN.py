import sys
import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import Adam

nb_classes = 10
batch_size = 50
nb_epoch=200


#handle data
f=open(sys.argv[1]+'/all_label.p','rb')
data = pickle.load(f)
data = np.array(data)

x_train = np.reshape(data,(-1,3,32,32)).astype('float32')
y_train = np.array(list( i for i in range(nb_classes) for j in range(500) ))

y_train = np_utils.to_categorical(y_train,nb_classes)

#x_train /= 255


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
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        shuffle=True,
        verbose=1,
)

model.save(sys.argv[2]+'_1')

#res = model.predict(x_train)
#loss,accuracy = model.evaluate(x_train,y_train)
#print("loss: ",loss," accuracy: ",accuracy)
#print([np.round(x) for x in res  ] )


