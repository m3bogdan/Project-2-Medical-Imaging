import numpy as np
import skimage.io as io
import skimage.segmentation as seg
import skimage.color as color
import matplotlib.pyplot as plt
import os
from skimage import img_as_ubyte
import skimage.filters as filters

def chan_vese_segmentation(folder_path_in, folder_path_out):
    
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        # Load the image
        image_path = folder_path_in + "/" + filename
        image = io.imread(image_path)
        # Ignore the alpha channel (e.g. transparency )
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = color.rgb2gray(image)
        image = filters.gaussian(image, sigma=2)

        # Perform segmentation using Chan-Vese active contour model
        segmentation = seg.chan_vese(image, mu = 0.01, max_num_iter = 1000)

        #Save the image in the new folder
        new_path = folder_path_out + "/" + filename
        io.imsave(new_path, img_as_ubyte(segmentation))




# Provide the image path and save path for the segmentation result
folder_path_in = "C:/Users/annam/Desktop/Globules/Resized"
folder_path_out = "C:/Users/annam/Desktop/Globules/Segmentation_2"

chan_vese_segmentation(folder_path_in, folder_path_out)