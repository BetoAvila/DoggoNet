from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Importing data
dn_input = np.load('/home/beto/Documents/projects/DoggoNet/DN-dataset.npz')

x_train = dn_input['X_train']
x_test = dn_input['X_test']
y_train = dn_input['y_train']
y_test = dn_input['y_test']

print('Training set shape:', x_train.shape)
print('Testing set shape:', x_test.shape)

# Selecting an example
# etiqueta = dn_input['y_train'][1200]
# print(etiqueta[0])
# foto = dn_input['X_train'][1200]
# foto = np.multiply(foto, 255).astype(np.int8)
# Image.fromarray(foto, 'RGB').show()

# convert class
num_classes = 4
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model initialization
model = keras.Sequential([keras.Input(shape=(240, 320, 3))])

# Adding logic to the model
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

add_dense_lyr(model, 100, 'tanh', 3)

# Ending model
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
add_dense_lyr(model, num_classes, 'softmax')
model.summary()

batch_size = 100
epochs = 12

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print('\n-------------------- MODEL COMPILED AND READY --------------------\n\n')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
