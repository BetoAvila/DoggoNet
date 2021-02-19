# importing all necesary libraries
from PIL import Image
import glob, os

# Resize images
path = "/home/beto/Documents/projects/DoggoNet/"
size = (320, 240)
i = 1
for infile in glob.glob(path + "perritos_org/*.jpeg"):  # for all jpg files
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)  # open them
    if 'jack' in file.lower():  # lowercase their names
        perrito = 'jack'
    elif 'volt' in file.lower():
        perrito = 'volt'
    elif 'luna' in file.lower():
        perrito = 'luna'
    elif 'katy' in file.lower():
        perrito = 'katy'

        # resize them and save them
    if im.height > im.width:
        im.transpose(Image.ROTATE_90).resize(size) \
            .save(fp=path + 'perritos/{}{}.jpeg'.format(perrito, i), format='jpeg')
    else:
        im.resize(size) \
            .save(fp=path + 'perritos/{}{}.jpeg'.format(perrito, i), format='jpeg')
    if i % 400 == 0: print(i // 400, 'out of', 10)
    i += 1