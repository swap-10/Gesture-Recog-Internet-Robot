# If using the csv dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def getdata(path):
    labels = []
    pixels = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in csvfile:
            row = row.split(',')
            labels.append(row[0])
            
            pix = row[1:]
            pix = np.array(pix)
            pix = pix.astype('uint8')
            pix = pix.reshape(28, 28, 1)
            # Here frame is of type numpy.ndarray
            blur = cv2.GaussianBlur(pix, (3, 3), 0)
            ret, pix = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pix = list(pix.flatten())
            pixels.append(pix)

    labels = np.array(labels)
    pixels = np.array(pixels)
    labels = labels.astype(int)
    pixels = pixels.astype(float)
    pixels = pixels.reshape((-1, 28, 28, 1))

    return pixels, labels

train_path = '.\sign_mnist_train.csv'
test_path = '.\sign_mnist_test.csv'

train_images, train_labels = getdata(train_path)
test_images, test_labels = getdata(test_path)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
'''
train_datagen = ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=(0.6, 1.0)
)
'''


train_datagen = ImageDataGenerator(rescale=1/255., zoom_range=0.4, rotation_range=30, horizontal_flip=True, brightness_range=(0.8, 1.0))
test_datagen = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow(train_images, train_labels, batch_size=200)
test_gen = test_datagen.flow(test_images, test_labels, batch_size=100)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu')])
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(len(np.unique(train_labels)) + 1, activation='softmax'))

NUM_EPOCHS = 25

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("Stopping training. Accuracy:", logs.get('accuracy') * 100, "%")
            self.model.stop_training=True


callback1 = myCallback()

log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_gen, epochs=NUM_EPOCHS, validation_data=test_gen, verbose=1, callbacks=[callback1, tensorboard_callback])
print(model.evaluate(test_gen, verbose=0))


def plot_metrics(metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel('NUM_EPOCHS')
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


plot_metrics('accuracy')
plot_metrics('loss')

model.save('cv_model.h5')
print("\nNow loading new model: ")
new_model = tf.keras.models.load_model('cv_model.h5')
print(model.summary())
print(new_model.evaluate(test_gen))