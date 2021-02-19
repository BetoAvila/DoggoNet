from pathlib import Path
from PIL import Image
import glob, os
import numpy as np

# Appending datapoints
path = "/home/beto/Documents/projects/DoggoNet/perritos/"  # location of resized images
dat = []
etq = []
i = 1
for infile in glob.glob(path + '*.jpeg'):  # for every jpeg file
    im = Image.open(infile)  # open every file
    im = np.asarray(im)  # convert it as np.array()
    im = np.divide(im, 255.0, dtype=np.float16)  # divide it by 255.0 to scale pixels values
    dat.append(im)  # and then apend it to the datos list
    if 'jack' in infile.lower():  # append the numeric label to each dog
        etq.append(np.asarray([0.0]))
    if 'luna' in infile.lower():
        etq.append(np.asarray([1.0]))
    if 'volt' in infile.lower():
        etq.append(np.asarray([2.0]))
    if 'katy' in infile.lower():
        etq.append(np.asarray([3.0]))
    if i % 400 == 0: print(i // 400)
    i += 1

# Creating np arrays from labels and datapoints
dat = np.asarray(dat, dtype=np.float16)
etq = np.asarray(etq, dtype=np.int8)

# Splitting data in trainig and test sets for further validation set split
# Proportions are train set 50%, validation set 35%, test set 15% 55,35,10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dat, etq, train_size=0.85, random_state=42)

# Save datasets as .npz file
np.savez_compressed('DN-dataset.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
