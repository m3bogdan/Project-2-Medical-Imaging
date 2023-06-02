import pandas as pd
import cv2
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import extract_features as feature

def load_model(model_file):
    with open(model_file, 'rb') as file:
        models = pickle.load(file)
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
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Accuracy of {name}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

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

def predict_from_image(models, image_path, best_model_name):
    all_features = extract_features(image_path)
    X = pd.DataFrame([all_features])  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = models[best_model_name]
    prediction = best_model.predict(X_scaled)
    probability = best_model.predict_proba(X_scaled)

    return prediction, probability

def main():
    models = load_model('group02_classifiers.sav')
    data = pd.read_csv('features.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_test = X.sample(5)  # assuming we evaluate on a small sample
    y_test = y.loc[X_test.index]

    model_scores = evaluate_classifier(models, X_test, y_test)
    cross_validation_evaluation(models, X, y)

    best_model_name = max(model_scores, key=lambda x: model_scores[x])
    print(f"Best model based on F1 Score is {best_model_name}")

    image_path = "PUT THE IMAGE PATH HERE"  # replace with your image path
    prediction, probability = predict_from_image(models, image_path, best_model_name)
    print("Prediction:", prediction)
    print("Probability:", probability)

if __name__ == '__main__':
    main()
