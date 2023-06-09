
import os
import pandas as pd
import numpy as np

# Default packages for the minimum example
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score #example for measuring performance
import pickle #for saving/loading trained classifiers
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


features = "PROJECT/fyp2023/features/features.csv"

df_merged = pd.read_csv(features)



# Split the data into training and testing sets
x = np.array(df_merged.drop(['img_id', 'diagnostic', 'patient_id'], axis=1))
y = df_merged['diagnostic']
patient_id = df_merged['img_id']


#Prepare cross-validation - images from the same patient must always stay together
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)
group_kfold.get_n_splits(x, y, patient_id)


#Different classifiers to test out
classifiers = [KNN(5), LR(max_iter= 5000), DTC()]
num_classifiers = len(classifiers)

      
acc_val = np.empty([num_folds,num_classifiers])
f1_val = np.empty([num_folds,num_classifiers])
precision = np.empty([num_folds,num_classifiers])
recall = np.empty([num_folds,num_classifiers])


for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
    
    x_train = x[train_index,:]
    y_train = y[train_index]
    x_val = x[val_index,:]
    y_val = y[val_index]
    
    for j, clf in enumerate(classifiers): 
        
        #Train the classifier
        clf.fit(x_train,y_train)
    
        #Evaluate your metric of choice (accuracy is probably not the best choice)
        acc_val[i,j] = accuracy_score(y_val, clf.predict(x_val))
        f1_val[i,j] = f1_score(y_val, clf.predict(x_val))
        precision[i, j] = precision_score(y_val, clf.predict(x_val), zero_division= 0)
        recall[i,j] = recall_score(y_val, clf.predict(x_val))
        #print(confusion_matrix(y, clf.predict(x_val)))
    
#Average over all folds
average_acc = np.mean(acc_val,axis=0) 
average_f1 = np.mean(f1_val, axis = 0)
average_precision = np.mean(precision, axis = 0)
average_recall = np.mean(recall, axis = 0)


print("############Classifier 1 - KNN:")
print('F1 score = {:.3f} '.format(average_f1[0]))   
print('Accuracy= {:.3f} '.format(average_acc[0]))
print('Precision = {:.3f} '.format(average_precision[0]))
print('Recall = {:.3f} '.format(average_recall[0]))

print("############ Classifier 2 - LR:")
print('F1 score = {:.3f} '.format(average_f1[1]))   
print('Accuracy= {:.3f} '.format(average_acc[1]))
print('Precision = {:.3f} '.format(average_precision[1]))
print('Recall = {:.3f} '.format(average_recall[1]))

print("############Classifier 3 - DTC:")  
print('F1 score = {:.3f} '.format(average_f1[2])) 
print('Accuracy= {:.3f} '.format(average_acc[2]))
print('Precision = {:.3f} '.format(average_precision[2]))
print('Recall = {:.3f} '.format(average_recall[2]))



#Let's say you now decided to use the 5-NN 
classifier = KNN(5)

#It will be tested on external data, so we can try to maximize the use of our available data by training on 
#ALL of x and y
classifier = classifier.fit(x,y)


#This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupXY_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))



