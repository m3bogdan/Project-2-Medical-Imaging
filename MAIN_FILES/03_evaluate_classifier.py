import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from 01_process_images import extract_features

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
    # Your feature extraction process goes here
    pass

def predict_from_image(models, image_path, best_model_name):
    features = extract_features(image_path)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    best_model = models[best_model_name]
    prediction = best_model.predict(features)
    return prediction

def main():
    models = load_model('models.pkl')
    data = pd.read_csv('features.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_test = X.sample(5)  # assuming we evaluate on a small sample
    y_test = y[X_test.index]

    model_scores = evaluate_classifier(models, X_test, y_test)
    cross_validation_evaluation(models, X, y)

    best_model_name = max(model_scores, key=model_scores.get)
    print(f"Best model based on F1 Score is {best_model_name}")

    image_path = "test_image.jpg"  # replace with your image path
    print("Prediction from image:", predict_from_image(models, image_path, best_model_name))

if __name__ == '__main__':
    main()
