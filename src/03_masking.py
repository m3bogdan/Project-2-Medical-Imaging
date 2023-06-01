
from skimage import io, util
import os
import numpy as np
import cv2



def mask_overlay(image_folder_path, mask_folder_path, folder_path_out):

    #Iterate through all the jpg and png files in the selected folder
    for filename in [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]:
        
        #Read in the image and the mask
        image_path = image_folder_path + "/" + filename
        mask_path = mask_folder_path + "/" + filename
        image = io.imread(image_path)
        mask = io.imread(mask_path)

        # Combine the image and the mask
        expanded_mask = mask[:, :, np.newaxis] #makes the mask 3D
        masked_image = cv2.bitwise_and(image, image, mask=expanded_mask)

        #Save the masked image in the new folder
        new_path = folder_path_out + "/" + filename
        io.imsave(new_path, util.img_as_ubyte(masked_image))

image_folder_path = "data/helpme/newimages"
mask_folder_path = "data/helpme/newmask"
folder_path_out = "data/helpme/color_mask"
mask_overlay(image_folder_path, mask_folder_path, folder_path_out)
