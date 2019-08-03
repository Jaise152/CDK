# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:41:01 2019

@author: 611916967
"""

import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
# train test split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
class Categorical_Imputer:
    """
    Imputing categorical data using the most frequent value
    """
    
    # instance attribute
    def __init__(self, strategy):
        self.strategy = strategy
        
    # instance method
    def fit_transform(self, df:'dataframe')->'dataframe':
        """
        Fill in missing categorical values using most frequent value
        """
        
        # instantiate CategoricalImputer
        imputer = CategoricalImputer()
        
        # convert array to dataframe
        df_filled = df.apply(lambda x: imputer.fit_transform(x), axis=0)
        
        # return filled dataframe
        return df_filled
        

df = pd.read_csv('C:\Personal\MTech\Books and Materials\Data Mining\Assignment\Assignment_BLR (1)\kidneyChronic.csv')
df.replace('?', np.nan, inplace=True)
#
numerical_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
categorical_columns = df.columns.drop('class').drop(numerical_columns)


# convert numerical data 
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
#
# check the number of unique values
df[categorical_columns].apply(lambda x: x.nunique(), axis=0)
#df['dm'].unique()
#df['cad'].unique()

df['dm']=df['dm'].str.strip()
df['cad']=df['cad'].str.strip()

df['class'] = df['class'].apply(lambda x: 1 if x=='ckd' else 0)

# define numerical imputer
num_imp = Imputer(missing_values=np.nan, strategy='median', axis=0)
# imputing on numerical data
df[numerical_columns] = num_imp.fit_transform(df[numerical_columns])

# define categorical imputer
cate_imputer = Categorical_Imputer('most_frequent')
# imputing on categorical data
df[categorical_columns] = cate_imputer.fit_transform(df[categorical_columns])
#
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0})

#Checking correlation between diffrent predictors
df2 = df.dropna(axis=0)
df2['class'].value_counts()
corr_df = df2.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = pyplot.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
pyplot.title('Correlations between different predictors')
pyplot.show()


#df.corr().to_csv('C:\Personal\MTech\Books and Materials\Data Mining\Assignment\Assignment_BLR (1)\kidneyChronic_processed.csv')


# load X and y
X = df.drop(columns=['class'])
y= df['class']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=21, stratify=y)

#Using Decision Tree Classification

Decision_tree_classification = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(Decision_tree_classification.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(Decision_tree_classification.score(X_test, y_test)))

probs = Decision_tree_classification.predict_proba(X_test) 
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

#Using Gaussian Naive Bayes
Gaussian_Naive_Bayes = GaussianNB()
Gaussian_Naive_Bayes.fit(X_train, y_train)
print('Accuracy of Gaussian Naive Bayes classifier on training set: {:.2f}'
     .format(Gaussian_Naive_Bayes.score(X_train, y_train)))
print('Accuracy of Gaussian Naive Bayes classifier on test set: {:.2f}'
     .format(Gaussian_Naive_Bayes.score(X_test, y_test)))
probs = Gaussian_Naive_Bayes.predict_proba(X_test) 
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

#Using SVM
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

#Using Random Forest Classifier
Random_forest_Classifier = RandomForestClassifier()
Random_forest_Classifier.fit(X_train, y_train)
print('Accuracy of Random forest Classifier classifier on training set: {:.2f}'
     .format(Random_forest_Classifier.score(X_train, y_train)))
print('Accuracy of Random forest Classifier classifier on test set: {:.2f}'
     .format(Random_forest_Classifier.score(X_test, y_test)))

models = ["Random_forest_Classifier","Decision_tree_classification","Gaussian_Naive_Bayes"]
#instantiate a roc_auc object
probs = Random_forest_Classifier.predict_proba(X_test) 
# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()


# Get numerical feature importances
feature_list = df.columns
importances = list(Random_forest_Classifier.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Set the style
pyplot .style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
pyplot .bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
pyplot .xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
pyplot .ylabel('Importance'); pyplot .xlabel('Variable'); pyplot .title('Variable Importances');
