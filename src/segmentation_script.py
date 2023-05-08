
from skimage import io
from skimage.filters import threshold_otsu
import os
from skimage.color import rgb2gray
from skimage import util

# Input the folder with the images and the name of the image
# this will be automated later on
folder_path = "C:/Users/annam/Desktop/Globules/Resized"
filename = "images.jpg"

# reads the image and converts it to grayscale 
image = io.imread(os.path.join(folder_path, filename))
grayscale = rgb2gray(image)

# Create a binary mask
thresh = threshold_otsu(grayscale)
binary = grayscale > thresh
inverted_img = util.invert(binary)

### DO NOT USE MATPLOTLIB FOR PLOTTING ###
# Show the image (will not be a part of the final code)
io.imshow(inverted_img)
io.show()

