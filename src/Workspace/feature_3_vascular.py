from skimage import io, color, img_as_ubyte
import numpy as np
import os
from skimage import exposure

folder_path_in = "C:/Users/annam/Desktop/Test/Resized"
 #Iterate through all the jpg and png files in the selected folder
for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
    
    #Read in the image
    image_path = folder_path_in + "/" + filename
    original = io.imread(image_path)

    # Ignore the alpha channel (e.g. transparency )
    if original.shape[-1] == 4:
        original = original[..., :3]


    # Enhance the red colour (idk if it does much)
    red_channel = original[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1.5)
    enhanced_image = original.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel

    hsv_img = color.rgb2hsv(enhanced_image) 

    lower_red = np.array([0/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red = np.array([25/360, 1, 1])  # Upper limit for red hue, saturation, and value

    lower_red2 = np.array([330/360, 40/100, 00/100])  # Lower limit for red hue, saturation, and value
    upper_red2 = np.array([1, 1, 1])  # Upper limit for red hue, saturation, and value

    mask = np.logical_or(
        np.logical_and(np.all(hsv_img >= lower_red, axis=-1), np.all(hsv_img <= upper_red, axis=-1)),
        np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1)))

    result = enhanced_image.copy()
    result[~mask] = 0
    folder_path_out = "C:/Users/annam/Desktop/Test/Diagnosis/"
    if result.max() > 0:
        print("Diagnosis: cancer, vascular pattern found")
                #Save the image in the new folder
        new_path = folder_path_out + "/Cancer/" + filename
        io.imsave(new_path, img_as_ubyte(original))
    else:
        print("Diagnosis: not a cancer")
        new_path = folder_path_out + "/No_cancer/" + filename
        io.imsave(new_path, img_as_ubyte(original))
   
