from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tkinter as tk
import os

# Ensuring execution on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

############################ Neural Net
# Importing data and model
path = '/home/beto/Documents/projects/DoggoNet/'
dn_input = np.load(path + 'DN-dataset.npz')
model = keras.models.load_model(path + 'model_03mar21_9832_h5')

from tensorflow.keras import layers


# Prediction functions
def predict_image(num):
    d = {0: 'Jack', 1: 'Luna', 2: 'Volt', 3: 'Katy'}
    i_real = dn_input['y_test'][num]
    foto = dn_input['X_test'][num]
    foto = np.multiply(foto, 255).astype(np.int8)
    foto = Image.fromarray(foto, 'RGB')
    foto = ImageTk.PhotoImage(foto)
    dog_photo_lbl.config(image=foto)
    dog_photo_lbl.image = foto
    dog_photo_lbl.grid(columnspan=2, column=1, row=0)
    result = model.predict(dn_input['X_test'][num].reshape(1, 225, 300, 3)).flatten()
    val = result.max()
    i_pred = np.where(np.isclose(result, val))[0][0]
    print(result[i_pred])
    return 'The neural net predicts with a\n{}% probability this is {}\nand the actual dog is {}'.format(
        round(result[i_pred] * 100, 2),
        d[i_pred], d[i_real[0]])


# Execute this on button click
def get_num():
    num = in_text.get(1.0, 'end-1c')
    if not num.isnumeric():
        label.config(text=wrn_text, bg='red')
        dog_photo_lbl.config(bg='white')
    elif (int(num) <= 0) | (int(num) >= 599):
        label.config(text=wrn_text, bg='red')
        dog_photo_lbl.config(bg='white')
    else:
        num = int(num)
        out_text = predict_image(num)
        label.config(text=out_text, bg='white')


############################ TKinter
intro_text = 'This is a convolutional neural network (CNN)\n' \
             'that predicts which dog is selected from a 600\n' \
             'photos set. This CNN has never seen before any\n' \
             'of the photos in the set, select a number from\n' \
             '0 to 599 to predict what doggo is that'
wrn_text = 'Select a number from 0 to 599\nto predict what doggo is that'
root = tk.Tk()  # Begining of GUI'The neural net predicts this is {}\nand the actual dog is {}'.format(i_pred, i_real)
root.configure(background='white')
root.resizable(width=False, height=False)
root.title('DoggoNet')

# Canvas creation
canvas = tk.Canvas(root, width=800, height=400, bg='white')  # variable to draw widgets
canvas.grid(columnspan=3)  # divide canvas in n identical invisible columns

# Adding a photo
photo = Image.open(path + 'DN_logo.png').resize((300, 225))
photo = ImageTk.PhotoImage(photo)
photo_label = tk.Label(image=photo)
photo_label.image = photo
photo_label.grid(column=0, row=0)

# Adding a text label
label = tk.Label(root, bg='white',
                 text=intro_text,
                 font=("Ubuntu", 16))
label.grid(column=0, row=1)

# Adding a text input
in_text = tk.Text(root, height=1, width=6, font=("Ubuntu", 16))
in_text.grid(column=0, row=2)

# Adding a predict button
predict_btn = tk.Button(root, text='Predict', height=2, width=9, font=('Ubuntu', 16), command=get_num)
predict_btn.grid(column=1, row=2)

# Adding a dog image label
dog_photo_lbl = tk.Label()

root.mainloop()  # End of GUI
