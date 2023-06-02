
from skimage import io, util, img_as_ubyte
from skimage.filters import threshold_otsu
import os
from skimage.color import rgb2gray
import skimage.filters as filters

def image_segment(folder_path_in, folder_path_out):

    #Iterate through all the jpg and png files in the selected folder
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        
        #Read in the image
        image_path = folder_path_in + "/" + filename
        original = io.imread(image_path)

        # Ignore the alpha channel (e.g. transparency )
        if original.shape[-1] == 4:
            original = original[..., :3]

        # Create a binary mask
        grayscale = rgb2gray(original)
        grayscale = filters.gaussian(grayscale, sigma=1)
        thresh = threshold_otsu(grayscale)
        binary = grayscale > thresh
        inverted_img = util.invert(binary)

        
        #Save the image in the new folder
        new_path = folder_path_out + "/" + filename
        io.imsave(new_path, img_as_ubyte(inverted_img))


#Specify where to pull images from and where to save them
folder_path_in = "C:/Users/annam/Desktop/Globules/Resized"
folder_path_out = "C:/Users/annam/Desktop/Globules/Segmentation"

image_segment(folder_path_in, folder_path_out)