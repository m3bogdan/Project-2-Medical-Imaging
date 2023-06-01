import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from 01_process_imgages import extract_features

def load_model(model_file):
    with open(model_file, 'rb') as file:
        models = pickle.load(file)
    return models

def evaluate_classifier(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-----------------------------------\n")

def cross_validation_evaluation(models, X, y):
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Accuracy of {name}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

def predict_from_image(models, image_path):
    features = extract_features(image_path)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Select the best model based on accuracy on the test set
    best_model_name = max(models, key=lambda x: models[x].score(features, y_test))
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

    evaluate_classifier(models, X_test, y_test)
    cross_validation_evaluation(models, X, y)
    image_path = "test_image.jpg"  # replace with your image path
    print("Prediction from image:", predict_from_image(models, image_path))

if __name__ == '__main__':
    main()
