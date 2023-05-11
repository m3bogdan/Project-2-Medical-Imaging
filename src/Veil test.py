import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb
from skimage import io
import os

#Iterate through all the jpg and png files in the selected folder
folder_path_in = "C:/Users/annam/Desktop/Veil/Resized/"
for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:

    #Read in the image
    
    
    image_path = folder_path_in + "/" + filename
    original = io.imread(image_path)
    # Ignore the alpha channel (e.g. transparency )
    if original.shape[-1] == 4:
        original = original[..., :3]


    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(original)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(original)
    ax[0].set_title("Original image")

    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")



    plt.tight_layout()
    plt.show()