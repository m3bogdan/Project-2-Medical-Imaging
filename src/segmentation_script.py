
from skimage import io
from skimage.filters import threshold_otsu
import os
from skimage.color import rgb2gray
from skimage import util


folder_path = "C:/Users/annam/Desktop/Globules/Resized"
image_path = "C:/Users/annam/Desktop/Globules/Resized/images.jpg"
filename = "images.jpg"

image = io.imread(os.path.join(folder_path, filename))
grayscale = rgb2gray(image)

thresh = threshold_otsu(grayscale)
binary = grayscale > thresh

### DO NOT USE MATPLOTLIB FOR PLOTTING ###
# io.imshow(binary)
# io.show()



inverted_img = util.invert(binary)

io.imshow(inverted_img)
io.show()