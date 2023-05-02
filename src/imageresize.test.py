
from PIL import Image
import os
from os import listdir

def getImageName(file_location):
    filename = file_location.split('/')[-1]
    location = file_location.split('/')[0:-1]
    filename = filename.split('.')
    filename[0] += "_resized"
    filename = '.'.join(filename)
    new_path = '/'.join(location) + '/' + filename
    return new_path


def image_resize(image_path):
    image = Image.open(image_path)
    new_height = 256
    new_width = int(new_height / image.height * image.width)
    resized_im = image.resize((new_width, new_height))
    size = resized_im.size
    name = getImageName(image_path)
    
    resized_im.save(name)
    #resized_im.show()
    return resized_im

def image_save(folder_dir, image):
    new_folder = folder_dir + "/" + "resized"
    image.save(new_folder)
    return None



def batch_resize(folder_dir):
    for image in os.listdir(folder_dir):
        x = folder_dir + "/" + image
        resized = image_resize(x)
        #image_save(folder_dir, resized)
    return "Resizing complete"


directory = 'C:/Users/annam/Desktop/Autumn'
image_example = "C:/Users/annam/Desktop/Autumn/pexels-eberhard-grossgasteiger-2310641.jpg"
print(batch_resize(directory))
#print(image_resize(image_example))