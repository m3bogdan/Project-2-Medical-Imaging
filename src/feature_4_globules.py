from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage import util
import os

# Create a list of image filenames
folder_path_in = "C:/Users/annam/Desktop/Test/Masked/"
filenames = []
for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
    filenames.append(filename)

# Iterate over images
for filename in enumerate(filenames):
    # Read the image
    image_path = folder_path_in + "/" + filename
    original = io.imread(image_path)
    # Ignore the alpha channel (e.g. transparency )
    if original.shape[-1] == 4:
        original = original[..., :3]



    # Preprocess the image
    image_gray = rgb2gray(original)
    inverted_image = util.invert(image_gray)

    # Detect blobs
    folder_path_out = "C:/Users/annam/Desktop/Test/Diagnosis/"
    blobs_doh = blob_log(inverted_image, min_sigma=0.8, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)
    if blob_amount > 600:
        print("Diagnosis: cancer, blobs found: ", blob_amount)
                #Save the image in the new folder
        new_path = folder_path_out + "/Cancer/" + filename
        io.imsave(new_path, img_as_ubyte(image_gray))
    else:
        print("Diagnosis: not a cancer, blobs found: ", blob_amount)
        new_path = folder_path_out + "/No_cancer/" + filename
        io.imsave(new_path, img_as_ubyte(image_gray))

################################ IF YOU WANT TO PLOT IT ################################

# # Create subplots based on the number of images
# num_images = len(filenames)
# num_cols = 2  
# num_rows = (num_images + num_cols - 1) // num_cols  

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# # Iterate over images and plot them
# for i, filename in enumerate(filenames):
#     # Read the image
#     image_path = folder_path_in + "/" + filename
#     original = io.imread(image_path)
#     # Ignore the alpha channel (e.g. transparency )
#     if original.shape[-1] == 4:
#         original = original[..., :3]

#     # Plot the image with blobs
#     ax = axes[i // num_cols, i % num_cols]
#     ax.set_title('Image' + filename + ': {} Blobs'.format(blob_amount))
#     ax.imshow(inverted_image, cmap='gray')
#     ax.axis('off')
    
#     for blob in blobs_doh:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
#         ax.add_patch(c)

# plt.tight_layout()
# plt.show()