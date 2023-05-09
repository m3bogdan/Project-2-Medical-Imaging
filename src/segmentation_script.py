
from skimage import io
from skimage.filters import threshold_otsu
import os
from skimage.color import rgb2gray
from skimage import util

# Set the path to the folder containing the images
folder_path = "C:/Users/annam/Desktop/Globules/Resized"
folder_path_out = "C:/Users/annam/Desktop/Globules/Segmentation"

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]

for filename in image_files:
    # Reads the image and removes the alpha channel if it exists
    image = io.imread(os.path.join(folder_path, filename))
    # Ignore the alpha channel (e.g. transparency )
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Create a binary mask
    grayscale = rgb2gray(image)
    thresh = threshold_otsu(grayscale)
    binary = grayscale > thresh
    inverted_img = util.invert(binary)
    
    # Save the images
    io.imsave(os.path.join(folder_path_out, 'mask_' + filename), inverted_img)