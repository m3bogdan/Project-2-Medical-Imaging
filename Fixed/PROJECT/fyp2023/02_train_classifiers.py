import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import pickle

features = "C:/Users/annam/Desktop/ITU/2nd_sem/02_First_Year_Project/2nd_project/Project-2_github_repo/Fixed/PROJECT/fyp2023/features/features.csv"

df_merged = pd.read_csv(features)

x = np.array(df_merged.drop(['img_id', 'diagnostic', 'patient_id'], axis=1))
y = df_merged['diagnostic']
patient_id = df_merged['img_id']

num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

classifiers = [KNN(5), LR(max_iter= 5000), DTC()]
num_classifiers = len(classifiers)

# Use PCA for the second round
use_pca = [False, True]

for pca in use_pca:
    print(f"Running with PCA: {pca}")
    if pca:
        pca_transformer = PCA(n_components=5)

    acc_val = np.empty([num_folds, num_classifiers])
    f1_val = np.empty([num_folds, num_classifiers])
    precision = np.empty([num_folds, num_classifiers])
    recall = np.empty([num_folds, num_classifiers])
    roc_auc = np.empty([num_folds, num_classifiers])
    
    for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):

        x_train, x_val = x[train_index,:], x[val_index,:]
        y_train, y_val = y[train_index], y[val_index]

        if pca:
            x_train = pca_transformer.fit_transform(x_train)
            x_val = pca_transformer.transform(x_val)

        for j, clf in enumerate(classifiers): 
            clf.fit(x_train, y_train)

            predictions = clf.predict(x_val)
            acc_val[i, j] = accuracy_score(y_val, predictions)
            f1_val[i, j] = f1_score(y_val, predictions)
            precision[i, j] = precision_score(y_val, predictions, zero_division= 0)
            recall[i, j] = recall_score(y_val, predictions)
            roc_auc[i, j] = roc_auc_score(y_val, predictions)

    average_acc = np.mean(acc_val,axis=0) 
    average_f1 = np.mean(f1_val, axis=0)
    average_precision = np.mean(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    average_roc_auc = np.mean(roc_auc, axis=0)

    # Print out the results
    for i, classifier_name in enumerate(["KNN", "LR", "DTC"]):
        print(f"############ Classifier {i+1} - {classifier_name}:")
        print(f'F1 score = {average_f1[i]:.3f}')
        print(f'Accuracy= {average_acc[i]:.3f}')
        print(f'Precision = {average_precision[i]:.3f}')
        print(f'Recall = {average_recall[i]:.3f}')
        print(f'ROC AUC = {average_roc_auc[i]:.3f}')

    # Plot ROC curve for best classifier
    best_classifier = np.argmax(average_f1 + average_precision + average_recall)
    clf = classifiers[best_classifier]
    clf.fit(x_train, y_train)
    y_score = clf.predict_proba(x_val)
    fpr, tpr, _ = roc_curve(y_val, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for ' + classifier_name)
    plt.legend(loc="lower right")
    plt.show()

    # Save the best model
    if pca:
        print(f"Best classifier is {classifier_name} with PCA")
        pickle.dump(clf, open(f"classifiers/classifier_pca_{classifier_name}.pkl", 'wb'))
    else:
        print(f"Best classifier is {classifier_name} without PCA")
        pickle.dump(clf, open(f"classifiers/classifier_{classifier_name}.pkl", 'wb'))

print("##############################################")

# Save the PCA transformer
if pca:
    pickle.dump(pca_transformer, open(f"classifiers/pca_transformer.pkl", 'wb'))
