from skimage import transform, io
import os
from skimage import img_as_ubyte



def image_resize(folder_path):
    for filename in [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]:
        image_path = folder_path + "/" + filename
        original = io.imread(image_path)
        new_height = int(256)
        new_width = int(new_height / original.shape[0] * original.shape[1])
        resized = transform.resize(original, (new_height, new_width))
        new_path = folder_path + "/Resized/" + filename
        io.imsave(new_path, img_as_ubyte(resized))

folder_path = "C:/Users/annam/Desktop/Globules"
image_resize(folder_path)






    