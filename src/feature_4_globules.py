from skimage import io, exposure
from skimage.color import rgb2gray
import matplotlib as plt

#To be adusted for iterating through a whole folder
image_path = "C:/Users/annam/Desktop/Globules/Segmentation/mask_images.jpg"

image = io.imread(image_path)
image_gray = rgb2gray(image)

hist, bins = exposure.histogram(image_gray)


plt.figure()
plt.bar(bins, hist, width=0.8, align='center')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()