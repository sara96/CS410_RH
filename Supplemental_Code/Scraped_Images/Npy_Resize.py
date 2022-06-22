from email.mime import image
from importlib.resources import path
import os
from unittest.mock import patch
import numpy as np  # to resize array
from PIL import Image  # to resize iamges


IMAGE_SIZE = 64
IMAGE_CHANNELS = 3  # RGB
IMAGE_DIR = input("which images folder would you like to resize: ")


images_path = IMAGE_DIR

training_data = []

print("resizing images...")
for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    # adding the resize images into the array
    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

# saving the images into binary file
print("saving...")
# np.save(IMAGE_DIR[:-1] +".npy", training_data)
np.save("NPY_Files/"+IMAGE_DIR[:]+".npy", training_data)
