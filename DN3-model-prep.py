from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import glob, os
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

dn_input = np.load('/home/beto/Documents/projects/DoggoNet/DN-dataset.npz')
print(dn_input.files)

x_train = dn_input['X_train']
x_test = dn_input['X_test']
y_train = dn_input['y_train']
y_test = dn_input['y_test']

print(x_train.shape)

etiqueta=dn_input['y_train'][1200]
print(etiqueta[0])

foto=dn_input['X_train'][1200]
foto = np.multiply(foto, 255).astype(np.int8)
Image.fromarray(foto, 'RGB').show()

# convert class
num_classes = 4
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define model
model = keras.Sequential(
    [
        keras.Input(shape=(240,320,3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
)
model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
dn_input.files