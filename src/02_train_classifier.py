import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import os

def load_data(features_file):
    data = pd.read_csv(features_file)
    return data

def split_data(data):
    X = data.drop('filename', axis=1)
    y = data['filename']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def train_classifier_without_PCA(X_train, y_train):
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'DTC': DecisionTreeClassifier(),
    }
        
    trained_classifiers = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            trained_classifiers[name] = clf

    return trained_classifiers

def train_classifier_with_PCA(X_train, y_train):
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'DTC': DecisionTreeClassifier(),
    }
    
    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    trained_classifiers = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            trained_classifiers[name] = clf

            clf_pca = clone(clf)
            clf_pca.fit(X_train_pca, y_train)
            trained_classifiers[f'PCA_{name}'] = clf_pca

    return trained_classifiers

def save_model(models, model_path):
    os.makedirs(model_path, exist_ok=True)
    for name, model in models.items():
        model_file = os.path.join(model_path, f'{name}.pickle')
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

def main():
    datacsv = load_data(r"C:\Users\serru\OneDrive\Documents\Project2\Project-2-Medical-Imaging\data\csv\full_data.csv")
    datafeatures = load_data(r"C:\Users\serru\Downloads\img\features.csv")
    X_train, X_test, y_train, y_test = split_data(data)
    model_path = r"C:\Users\serru\Downloads\img\Models"
    models = train_classifier_without_PCA(X_train, y_train)
    save_model(models, model_path)
    print("Models saved to disk.")

if __name__ == '__main__':
    main()