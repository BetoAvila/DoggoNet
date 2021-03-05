from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tkinter as tk

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

############## Neural Net
# Importing data and model
path = '/home/beto/Documents/projects/DoggoNet/'
dn_input = np.load(path + 'DN-dataset.npz')
model = keras.models.load_model(path + 'model_03mar21_9900')


# Definition
def test_image(num):
    d = {0: 'Jack', 1: 'Luna', 2: 'Volt', 3: 'Katy'}
    etiqueta = dn_input['y_test'][num]
    foto = dn_input['X_test'][num]
    foto = np.multiply(foto, 255).astype(np.int8)
    result = model.predict(dn_input['X_test'][num].reshape(1, 225, 300, 3)).flatten()
    val = result.max()
    i = np.where(np.isclose(result, val))[0][0]
    print('Neural network predicts this is {}\nShowing image to compare'.format(d[i]))
    Image.fromarray(foto, 'RGB').show()


############## TKinter
root = tk.Tk()  # Begining of GUI
root.title('DoggoNet')

# Canvas creation
canvas = tk.Canvas(root, width=480, height=480)  # variable to draw widgets
canvas.grid(columnspan=2)  # divide canvas in 3 identical invisible columns

# Adding a photo
photo = Image.open(path + 'DN_logo.png')
photo = ImageTk.PhotoImage(photo)
photo_label = tk.Label(image=photo)
photo_label.image = photo
photo_label.grid(columnspan=3, row=0)

# Adding a label
label = tk.Label(root,
                        text='Select a photo by typing a number\nbetween 0 and 599 to test DoggoNet',
                        font=("Ubuntu", 20))
label.grid(columnspan=3, row=1)

# Adding a button
predict_btn = tk.Button(root, text='Predict', height=3,
                        width=12, font=('Arial', 16))
predict_btn.grid(column=2, row=2)

# Adding a text input
in_text = tk.Text(root, height=1, width=6, font=("Ubuntu", 20))
in_text.grid(column=1, row=2)

root.mainloop()  # End of GUI
