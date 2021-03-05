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
def predict_image(num):
    d = {0: 'Jack', 1: 'Luna', 2: 'Volt', 3: 'Katy'}
    i_real = dn_input['y_test'][num]
    foto = dn_input['X_test'][num]
    foto = np.multiply(foto, 255).astype(np.int8)
    result = model.predict(dn_input['X_test'][num].reshape(1, 225, 300, 3)).flatten()
    val = result.max()
    i_pred = np.where(np.isclose(result, val))[0][0]
    return 'The neural net predicts this is {}\nand the actual dog is {}'.format(d[i_pred], d[i_real[0]])
    # Image.fromarray(foto, 'RGB').show()


############## TKinter
root = tk.Tk()  # Begining of GUI'The neural net predicts this is {}\nand the actual dog is {}'.format(i_pred, i_real)
root.configure(background='white')
root.resizable(width=False, height=False)
root.title('DoggoNet')

# Canvas creation
canvas = tk.Canvas(root, width=1080, height=480, bg='white')  # variable to draw widgets
canvas.grid(columnspan=4)  # divide canvas in 3 identical invisible columns

# Adding a photo
photo = Image.open(path + 'DN_logo.png')
photo = ImageTk.PhotoImage(photo)
photo_label = tk.Label(image=photo)
photo_label.image = photo
photo_label.grid(column=0, row=0)

# Adding a label
label = tk.Label(root, bg='white',
                 text='Select a photo by typing a number\nbetween 0 and 599 to test DoggoNet',
                 font=("Ubuntu", 20))
label.grid(column=0, row=1)

# Adding a text input
in_text = tk.Text(root, height=1, width=6, font=("Ubuntu", 18))
in_text.grid(column=0, row=2)


def get_num():
    num = in_text.get(1.0, 'end-1c')
    if not num.isnumeric():
        label.config(text=num, bg='red')
    else:
        num = int(num)
        out_text = predict_image(num)
        label.config(text=out_text, bg='white')


# Adding a button
predict_btn = tk.Button(root, text='Predict', height=2, width=9, font=('Ubuntu', 16), command=get_num)
predict_btn.grid(column=1, row=2)
root.mainloop()  # End of GUI
