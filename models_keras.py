#Import the necassary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical, plot_model
from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers
from keras.callbacks import ModelCheckpoint
from load_and_explore_data import X_train, y_train, X_valid, y_valid, X_test, y_test

#Reshape the data fot the convolutional neural network
#X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
#X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 3).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')

#Reshape the data for the multi layer perceptron
X_train = X_train.reshape(X_train.shape[0],32*32*3).astype('float32')
X_valid = X_valid.reshape(X_valid.shape[0],32*32*3).astype('float32')
X_test = X_test.reshape(X_test.shape[0],32*32*3).astype('float32')

#Normalize the data
X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

#Convert to categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

#Define the number of classes
num_classes = y_test.shape[1] # number of categories

#Define the baseline, a multi layer perceptron
def multi_layer_perceptron():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    #plot_model(model, to_file="model.png", show_shapes=True)
    
    #Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

#Define the LeNet model
def lenet():
    model = Sequential()
    model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(AveragePooling2D())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(43, activation = 'softmax'))
    model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

#Define the AlexNet model
def alexnet():
    model = Sequential()
    model.add(Conv2D(filters=96, activation = 'relu', input_shape=(32,32,3), kernel_size=(3,3), padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256, activation = 'relu', kernel_size=(3,3), padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=384, activation = 'relu', kernel_size=(3,3), padding = 'same'))
    model.add(Conv2D(filters=384, activation = 'relu', kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Conv2D(filters=256, activation = 'relu', kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(43, activation = 'softmax'))
    model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

#Define the VGGNet model
def vggnet():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.summary()
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum =0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Define a convolutional neural network
def convolutional_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 1)))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 1)))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    #Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

#Build the model
model = multi_layer_perceptron()

#Fit the model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=200, verbose=2,shuffle=True, callbacks=[ModelCheckpoint('model.h5',save_best_only=True)])

# summarize history for acc
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

#Another way to evaluate the model
y_pred = model.predict_classes(X_test)
y_pred = to_categorical(y_pred)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))

'''
predictions = model.predict(x_test)
print('First prediction:', predictions[0])
 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

#Third way to evaluate the model
y_prob = model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)
y_classes = to_categorical(y_classes)
accuracy = acc = np.sum(y_classes==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(accuracy))