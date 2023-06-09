"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Import our own file that has the feature extraction functions
from extract_features import extract_features



#-------------------
# Main script
#-------------------


#Where is the raw data
file_data = "PROJECT/fyp2023/data/metadata.csv"
path_image = "PROJECT/fyp2023/data/images/imgs_part_1"

#Where we will store the features
file_features = "PROJECT/fyp2023/features/features.csv"


#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)
df['diagnostic'] = df['diagnostic'].map({'BCC': 1, 'MEL': 1, 'SCC': 1, 'ACK': 0, 'NEV': 0, 'SEK': 0})


def extract_features_folder(path_image):
    features = []

    for filename in os.listdir(path_image):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(path_image, filename)
            image = io.imread(image_path)

            # Ignore the alpha channel (e.g. transparency)
            if image.shape[-1] == 4:
                image = image[..., :3]

            image_features = extract_features(image)
            image_features["img_id"] = filename
            features.append(image_features)

    features_df = pd.DataFrame(features)
    return features_df

# Merge the features DataFrame with the diagnostic column from the original DataFrame
df_merged = pd.merge(df[['img_id', 'diagnostic', 'patient_id']], extract_features_folder(path_image), on='img_id', how='inner')   
df_merged.to_csv(file_features, index=False)  



