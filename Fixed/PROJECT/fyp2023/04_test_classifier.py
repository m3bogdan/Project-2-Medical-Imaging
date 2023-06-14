import pickle
import pandas as pd
import os
from skimage import io
from extract_features import extract_features
from sklearn.metrics import f1_score

def extract_features_folder(path_image):
    features = []

    for filename in os.listdir(path_image):
        if filename.endswith(('.jpg', '.png')):
            img_id = os.path.splitext(filename)[0]  # Remove file extension
            image_path = os.path.join(path_image, filename)
            image = io.imread(image_path)

            image_features = extract_features(image)
            image_features["img_id"] = img_id
            features.append(image_features)

    features_df = pd.DataFrame(features)
    return features_df

def open_picke(picked_path):

    with open(picked_path, 'rb') as file:
        model = pickle.load(file)
    return model

def create_df(image_folder,csv_path):
    df = pd.read_csv(csv_path)
    features = extract_features_folder(image_folder)
    df_merged = pd.merge(df, features, on='img_id', how='inner')
    return df_merged

def test(pickled_path,image_folder,csv_path):
    picked_file = open_picke(pickled_path)    
    data = create_df(image_folder,csv_path)
    print(data)
    X = data.drop(['diagnostic', 'img_id'], axis=1)
    y = data['diagnostic']
    y_predict =  picked_file.predict(X)
    print(y_predict)
    f1_sc = f1_score(y,y_predict)
    print(f"the f1 score is. {f1_sc}")

    return f1_sc

pickled_path = r"Fixed\PROJECT\fyp2023\models\group02_classifier.sav"
image_folder = r"C:\Users\45911\Desktop\DS\Semester2\First_year_project\Project2\Pictures\data\TestMixed"
csv_path = r"Fixed\PROJECT\fyp2023\csv_datatypes\Mixed.csv"
test(pickled_path,image_folder,csv_path)


