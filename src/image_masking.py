
from skimage import io
import numpy as np
import cv2
from skimage import morphology

#put the image and mask path
image_path = "C:/Users/annam/Desktop/Globules/Resized/images.jpg"
mask_path = "C:/Users/annam/Desktop/Globules/Segmentation/mask_images.jpg"

#REad the image
image = io.imread(image_path)
mask = io.imread(mask_path)

expanded_mask = mask[:, :, np.newaxis] #makes the mask 3D
masked_image = cv2.bitwise_and(image, image, mask=expanded_mask)

io.imshow(masked_image)
io.show()