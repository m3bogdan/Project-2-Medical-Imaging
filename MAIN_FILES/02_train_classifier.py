import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
import pickle

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

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        trained_classifiers[name] = clf

        clf_pca = clone(clf)
        clf_pca.fit(X_train_pca, y_train)
        trained_classifiers[f'PCA_{name}'] = clf_pca

    return trained_classifiers

def save_model(models, model_file):
    with open(model_file, 'wb') as file:
        pickle.dump(models, file)


def main():
    data = load_data('features/features.csv')
    X_train, X_test, y_train, y_test = split_data(data)
    models = train_classifier_without_PCA(X_train, y_train)
    #models2 = train_classifier_with_PCA(X_train, y_train)
    #models.update(models2)
    save_model(models, 'group02_classifiers.sav')

if __name__ == '__main__':
    main()
