"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from extract_features import all

# Specify where to pull images from and where to save them
folder_path_in = r'C:\Users\serru\OneDrive\Documents\Project2\Project-2-Medical-Imaging\src\MAIN_FILES\MAIN_DATA'
folder_path_out = r'C:\Users\serru\OneDrive\Documents\Project2\Project-2-Medical-Imaging\src\MAIN_FILES\MAIN_DATA'

def image_resize(folder_path_in, folder_path_out):

    #Iterate through all the jpg and png files in the selected folder
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        
        #Read in the image
        image_path = folder_path_in + "/" + filename
        original = io.imread(image_path)

        # Ignore the alpha channel (e.g. transparency )
        if original.shape[-1] == 4:
            original = original[..., :3]

        #Resize the image (preserving the proportions)
        new_height = int(256)
        new_width = int(new_height / original.shape[0] * original.shape[1])
        resized = transform.resize(original, (new_height, new_width))

        #Save the image in the new folder
        new_path = folder_path_out + "/" + filename
        io.imsave(new_path, img_as_ubyte(resized))



#Specify where to pull images from and where to save them
folder_path_in = "C:/Users/annam/Desktop/Vascular/Original"
folder_path_out = "C:/Users/annam/Desktop/Vascular/Resized"
image_resize(folder_path_in, folder_path_out)