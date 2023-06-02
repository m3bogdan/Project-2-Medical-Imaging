import pandas as pd
import cv2
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import extract_features as feature

def load_model(model_folder):
    models = {}
    for filename in os.listdir(model_folder):
        model_file = os.path.join(model_folder, filename)
        with open(model_file, 'rb') as file:
            loaded_model = pickle.load(file)
        model_name = os.path.splitext(filename)[0]
        models[model_name] = loaded_model
    return models

def evaluate_classifier(models, X_test, y_test):
    model_scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        model_scores[name] = f1
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"F1 Score: {f1}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-----------------------------------\n")
    return model_scores

def cross_validation_evaluation(models, X, y):
    for name, model in models.items():
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
        print(f"Class Distribution for {name}:")
        for train_index, test_index in sss.split(X, y):
            train_classes = np.unique(y[train_index])
            test_classes = np.unique(y[test_index])
            print(f"Train Classes: {train_classes}")
            print(f"Test Classes: {test_classes}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"Model: {name}")
            print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            print(f"F1 Score: {f1}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("-----------------------------------\n")

def extract_features(image_path):
    image = cv2.imread(image_path)
    
    features = {}
    features['filename'] = os.path.basename(image_path)
    features['pigment_network_coverage'] = feature.measure_pigment_network(image)
    features['blue_veil_pixels'] = feature.measure_blue_veil(image)
    features['vascular_pixels'] = feature.measure_vascular(image)
    features['globules_count'] = feature.measure_globules(image)
    features['streaks_irregularity'] = feature.measure_streaks(image)
    features['irregular_pigmentation_coverage'] = feature.measure_irregular_pigmentation(image)
    features['regression_pixels'] = feature.measure_regression(image)

    return features

def predict_from_image(models, image_path):
    all_features = extract_features(image_path)
    X = pd.DataFrame([all_features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = {}
    probabilities = {}

    for model_name, model in models.items():
        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled)
        predictions[model_name] = prediction
        probabilities[model_name] = probability

    return predictions, probabilities

def main(model_folder):
    models = load_model(model_folder)
    
    data = pd.read_csv(r"C:\Users\serru\Downloads\img\features.csv")
    X = data.drop('filename', axis=1)
    y = data['filename']
    X_test = X.sample(n=5)  # assuming we evaluate on a small sample
    y_test = y.loc[X_test.index]

    model_scores = evaluate_classifier(models, X_test, y_test)
    cross_validation_evaluation(models, X, y)

    best_model_name = max(model_scores, key=lambda x: model_scores[x])
    print(f"Best model based on F1 Score is {best_model_name}")

    image_folder = r"data/images/Masks/Color_mask/Test"  # replace with your image path
    for filename in os.listdir(image_folder):
        # Construct the full path to the image file
        image_path = os.path.join(image_folder, filename)

        # Predict for the current image
        predictions, probabilities = predict_from_image(models, image_path)

        # Print the results for each model
        for model_name, prediction in predictions.items():
            print("Model:", model_name)
            print("Image:", filename)
            print("Prediction:", prediction[model_name])
            print("Probability:", probabilities[model_name])
            print("-----------------------------------")


if __name__ == '__main__':
    model_folder = r"C:\Users\serru\Downloads\img\Models"
    main(model_folder)
