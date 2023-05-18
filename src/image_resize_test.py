#imports
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage import util
from skimage import transform


def image_resize(folder_path, resized_folder_path): 

    # Define the target size 
    target_size = (256, 256)

    # Loop over the image file names and resize each image
    for filename in [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]:
        # Load the image from file
        image = io.imread(os.path.join(folder_path, filename))
        # Resize the image to the target size
        resized_image = transform.resize(image, target_size)
        # Save the resized image to file
        io.imsave(os.path.join(folder_path + "/Resized", filename), resized_image)


folder_path = "C:/Users/annam/Desktop/Globules"
folder_path_out = "C:/Users/annam/Desktop/Globules/Resized"

image_resize(folder_path, folder_path_out)