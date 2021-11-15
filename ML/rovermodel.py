# If using own dataset using createdata.py or otherwise

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def transf_image(image):
    blur = cv2.GaussianBlur(image.astype('uint8'), (31,31), 0)
    ret, image = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = image.astype(float)
    image = image.reshape(150, 150, 1)
    return image

train_datagen = ImageDataGenerator(
    preprocessing_function = transf_image,
    rescale = 1/255.,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.3,
    brightness_range=(0.6, 1.2),
    horizontal_flip=True,
    fill_mode = 'wrap'
)

test_datagen = ImageDataGenerator(preprocessing_function=transf_image, rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    '.\dataset\\train',
    target_size = (150, 150),
    color_mode='grayscale',
    batch_size=10,
    class_mode='sparse',
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    '.\dataset\\test',
    target_size=(150, 150),
    color_mode='grayscale',
    batch_size=10,
    class_mode='sparse',
    shuffle=True
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("Stopping training. Accuracy: ", logs.get('accuracy') * 100, "%")
            self.model.stop_training = True

callback1 = myCallback()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit/', histogram_freq=1)

inputs = tf.keras.layers.Input(shape=(150, 150, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1))(inputs)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(28, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='newds_model1')

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=10, validation_data=test_gen, callbacks=[callback1], verbose=1)

print(model.evaluate(test_gen, verbose=0))

def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel('NUM_EPOCHS')
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

plot_metrics(history, 'accuracy')
plot_metrics(history, 'loss')

model.save('newds_model.h5')
tf.keras.backend.clear_session()
new_model = tf.keras.models.load_model('newds_model.h5')
print(new_model.summary())
print(new_model.evaluate(test_gen, verbose=0))
