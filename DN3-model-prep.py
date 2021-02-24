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


# Defining building blocks
def add_conv_lyr(num_neurons, k_size, act_func, times=1):
    for i in range(times):
        model.add(layers.Conv2D(num_neurons, kernel_size=k_size, activation=act_func))


def add_max_pool_lyr(p_size, times=1):
    for i in range(times):
        model.add(layers.MaxPooling2D(pool_size=p_size))


def add_dense_lyr(num_neurons, act_func, times=1):
    for i in range(times):
        model.add(layers.Dense(num_neurons, activation=act_func))


# Adding logic to the model
add_conv_lyr(64, (3, 3), 'relu', 2)
add_max_pool_lyr((2, 2))
add_conv_lyr(64, (3, 3), 'relu', 2)
add_max_pool_lyr((2, 2))

add_dense_lyr(100, 'tanh', 3)

# Ending model
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
add_dense_lyr(num_classes, 'softmax')
model.summary()

batch_size = 100
epochs = 12

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print('\n-------------------- MODEL COMPILED AND READY --------------------\n\n')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
