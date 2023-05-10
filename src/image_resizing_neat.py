from skimage import transform, io
import os
from skimage import img_as_ubyte



def image_resize(folder_path_in, folder_path_out):

    #Iterate through all the jpg and png files in the selected folder
    for filename in [f for f in os.listdir(folder_path_in) if f.endswith('.jpg') or f.endswith('.png')]:
        
        #Read in the image
        image_path = folder_path_in + "/" + filename
        original = io.imread(image_path)

        #Resize the image (preserving the proportions)
        new_height = int(256)
        new_width = int(new_height / original.shape[0] * original.shape[1])
        resized = transform.resize(original, (new_height, new_width))

        #Save the image in the new folder
        new_path = folder_path_out + "/" + filename
        io.imsave(new_path, img_as_ubyte(resized))



#Specify where to pull images from and where to save them
folder_path_in = "C:/Users/annam/Desktop/Globules/Original"
folder_path_out = "C:/Users/annam/Desktop/Globules/Resized"
image_resize(folder_path_in, folder_path_out)






    