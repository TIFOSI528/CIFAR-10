# with ImageDataGenerator, BatchNormalization, regularizers.l2(0.01)
​
import keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.utils import np_utils
from keras.datasets import cifar10
​
# load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
​

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= 255  
X_test /= 255
​
batch_size = 128
epochs = 100
heights = 32
widths = 32
nb_classes = 10
​
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
​
input_tensor = Input((heights, widths, 3))
x = input_tensor
​
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
​
​
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
​
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
​
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
​
x = Flatten()(x)
​
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
​
predictions = Dense(10, activation='softmax')(x)
​
model = Model(inputs=input_tensor, outputs=predictions)
​
model.summary()
​
model.compile(optimizer = 'adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
​
# use ImageDataGenerator to generate batches of tensor image data with real-time data augmentation 

train_datagen = ImageDataGenerator(
    featurewise_center=True,              # Set input mean to 0 over the dataset, feature-wise.
    featurewise_std_normalization=True,         # Divide inputs by std of the dataset, feature-wise.
    rotation_range=20,          # Degree range for random rotations.
    shear_range=0.2,            # Shear Intensity (Shear angle in counter-clockwise direction as radians)
    zoom_range=0.2,             # Range for random zoom.
    fill_mode='nearest',        # Points outside the boundaries of the input are filled according to the given mode.
    horizontal_flip=True,       # Randomly flip inputs horizontally.
    )       
​
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    )
​
train_datagen.fit(X_train)
test_datagen.fit(X_test)
​
# fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)/batch_size, 
                        validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size), 
                        validation_steps = len(X_test)/batch_size, epochs=epochs)


# visualize the training process
​
import matplotlib.pyplot as plt
​
# list the data in history
print(history.history.keys())
​
# summarize history of accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
​
# summarize history of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
