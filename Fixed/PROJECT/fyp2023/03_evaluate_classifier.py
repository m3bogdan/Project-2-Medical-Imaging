import pickle #for loading your trained classifier
from skimage import io

from extract_features import extract_features #our feature extraction



# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(combined_image):
    
     #Resize the image etc, if you did that during training
    
     #combined_image = apply_mask(img, mask)


     #Extract features (the same ones that you used for training)
     X = list(extract_features(combined_image).values())
     X = np.array(X).reshape(1, -1)   
     
     # Load the trained classifier
     classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))

     # Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(X)
     pred_prob = classifier.predict_proba(X)

     # Get the probability for the predicted class
     predicted_class_prob = pred_prob[0, pred_label[0]]

     print('predicted label is', pred_label)
     print('predicted probability for the predicted class is', predicted_class_prob)
     
     return pred_label, predicted_class_prob

from skimage import io
import numpy as np

def apply_mask(image, mask):



    # Ensure image and mask have the same shape
    assert image.shape[:2] == mask.shape[:2], "Image and mask sizes don't match"

    # Create a copy of the image and apply the mask
    masked_image = np.copy(image)
    masked_image[np.where(mask == 0)] = 0

    return masked_image

import os
combined_path = "C:/Users/annam/Desktop/Binary masks/Resized/PAT_109_868_723.png"
combined = io.imread(combined_path)
#masked_image = apply_mask(image_path, mask_path)
test_path = "C:/Users/annam/Desktop/Binary masks/Resized"
for filename in os.listdir(test_path):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(test_path, filename)
        image = io.imread(image_path)

        # Ignore the alpha channel (e.g. transparency)
        if image.shape[-1] == 4:
            image = image[..., :3]

        classify(image)