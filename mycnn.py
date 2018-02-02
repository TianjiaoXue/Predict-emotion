import keras
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.utils import np_utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os


# input image dimensions
img_rows, img_cols = 100, 100

# number of channels
img_channels = 1

path1 = './emotion/'    #path of folder of images
path2 = './resize/'  #path of folder to save images
# returns a list containing the names of the entries in the directory given by path
listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples
for file in listing:
    if not file.startswith('.'):
        im = Image.open(path1 + '/' + file)
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L')
        gray.save(path2 +'/' +  file, "JPEG")
imlist = os.listdir(path2)
im1 = array(Image.open('resize' + '/'+ imlist[0])) # open one image to get size
# create matrix to store all flattened images
immatrix = array([array(Image.open('resize'+ '/' + im2)).flatten() for im2 in imlist],'f')
# shape : Shape of the new array,
# dtype : datatype
label=np.ones((num_samples,),dtype = int)
label[0:100]=0
label[101:177]=1
label[178:279]=2
label[280:349]=3
label[350:419]=4
label[420:524]=5
label[525:639]=6
label[640:805]=7
#make sure each train set are same by setting ramdom_state
data,Label = shuffle(immatrix,label)
train_data = [data,Label]

# images
print (train_data[0].shape)
# lables
print (train_data[1].shape)


#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 8
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# X: image martrix--train data
# y: label matrix-- train target
(X, y) = (train_data[0],train_data[1])

# ramdom_state : seed of ramdom number--make sure you always get same array for each train
# 20% of samples are testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 1 - number of features, there is only one feature for this datset
X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,1)

# converted to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize to 0-1
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# building layers
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(200, 200, 1)))
#model.add(BatchNormalization())
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.summary()
# corespond loss function for softmax is categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])

# verbose = 0 - silent; 1 - progress bar; 2 8- online per epoch
# train set = steps * batch_size
# validation_data: tuple (x_val, y_val) on which to evaluate the loss and any model metrics at the end of each epoch.

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=(X_test, Y_test))

# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_split=0.2)


# Returns the loss value & metrics values for the model in test mode.
test_score = model.evaluate(X_train,Y_train,verbose=2)


score = model.evaluate(X_test, Y_test, verbose=2)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Confusion Matrix of prediction of test set
from sklearn.metrics import classification_report,confusion_matrix

# Generates output predictions for the input samples.
# Y_pred:A Numpy array of predictions.
Y_pred = model.predict(X_test)

print('Y prediction ')
print(Y_pred)

print('y prediction')
#np.argmax(): Returns the index of the maximum values along an axis.
#axis =1 : along row
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

target_names = ['class 0(Amusement)', 'class 1(Anger)', 'class 2(Awe)','class3(Contentment)','class4(Disgust)','class5(Excitement)','class6(Fear)','class7(sad)']

print "Classification Report:"
# y_true : 1d array-like, or label indicator array / sparse matrix Ground truth (correct) target values.
# Accuarcy = (TP+TN)/(TP+FN+FP+TN)
# Recall = TP/(TP+FN)
# precision = TP/(TP+FP)
# f1-score = 2*P*R/(P+R)
# p-percision; R-Recall
# support: The number of occurrences of each label in y_true.
# with higher value, the model is better
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

print "Confusion matrix: "
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights
fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)

# Loading weights
fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)

