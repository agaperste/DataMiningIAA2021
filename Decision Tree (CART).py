#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:45:32 2020

@author: zhanyina

@citations:
    https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/
    https://towardsdatascience.com/decision-tree-algorithm-for-multiclass-problems-using-python-6b0ec1183bf5
    https://towardsdatascience.com/decision-tree-classifier-and-cost-computation-pruning-using-python-b93a0985ea77
    https://towardsdatascience.com/understanding-decision-tree-classification-with-scikit-learn-2ddf272731bd
"""


# importing necessary libraries 
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt
from IPython.display import display, Image
import missingno as msno_plot
import seaborn as sns
import pandas as pd
import numpy as np
import os, time
import pydotplus
import graphviz
from pydot import graph_from_dot_data

%matplotlib inline
sns.set(color_codes=True)

start = time.time()

# reading the data
os.chdir("/Users/zhanyina/Documents/MSA/AA502 Analytics Methods and Applications I/Data Mining/Asgnmts/2")
df_raw = pd.read_csv("bankData.csv")
pd.set_option('display.max_columns', None)
df_raw.head(5)
list(df_raw.columns) 
len(df_raw)
# getting a list of outcomes
uniqueProduct = df_raw["next.product"].unique()
uniqueProduct
del df_raw['y']

# Quick descriptive statistics of the data
df_raw.describe().transpose().round(2)

# check for missing values --> none
plt.title('#Non-missing Values by Columns')
msno_plot.bar(df_raw);
    
# pairplot
sns.pairplot(df_raw);

# heatmap --> correlated variables
plt.figure(figsize=(10,8))
sns.heatmap(df_raw.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()

# splitting data into training and test set for independent attributes
X_train_0, X_test_0, y_train, y_test = train_test_split(
    df_raw.drop('next.product',axis=1), df_raw['next.product'], test_size=.3, random_state=22)
X_train_0.shape, X_test_0.shape
X_train_0.columns


# def dummy_convert(data):
#     data['job'] = pd.get_dummies(data.job)
#     data['marital'] = pd.get_dummies(data.marital)
#     data['education'] = pd.get_dummies(data.education)
#     data['default'] = pd.get_dummies(data.default)
#     data['housing'] = pd.get_dummies(data.housing)
#     data['loan'] = pd.get_dummies(data.loan)
#     data = data.fillna(-999)
#     return data

# X_train=dummy_convert(X_train_0)
# X_test=dummy_convert(X_test_0)

# # !WRONG? cannot use LabelEncoder on X's, use only on Y!
# le = preprocessing.LabelEncoder()
# def label_enconder(data):
#     le = preprocessing.LabelEncoder()
#     data['job'] = le.fit_transform(data.job)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("job: ", le_name_mapping)
    
#     data['marital'] = le.fit_transform(data.marital)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("marital: ", le_name_mapping)
    
#     data['education'] = le.fit_transform(data.education)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("education: ", le_name_mapping)
    
#     data['default'] = le.fit_transform(data.default)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("default: ", le_name_mapping)
    
#     data['housing'] = le.fit_transform(data.housing)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("housing: ", le_name_mapping)
    
#     data['loan'] = le.fit_transform(data.loan)
#     le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#     print("loan: ", le_name_mapping)
    
#     data = data.fillna(-999)
#     return data

# X_train=label_enconder(X_train_0)
# X_test=label_enconder(X_test_0)

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def one_hot_encoder(data):
    enc = preprocessing.OneHotEncoder()
    data['job'] = enc.fit([['admin', 1], ['blue-collar', 2], ['entrepreneur', 3],
                            ['housemaid', 4], ['management', 5], ['retired', 6], ['self-employed', 7],
                            ['services', 8], ['student', 9], ['technician', 10], 
                            ['unemployed', 11], ['unknown', 12]]).transform([['admin', 1], ['blue-collar', 2], ['entrepreneur', 3],
                ['housemaid', 4], ['management', 5], ['retired', 6], ['self-employed', 7],
                ['unemployed', 11], ['unknown', 12]])
    data['marital'] = enc.fit([['divorced', 1], ['married', 2], ['single', 3], 
                                ['unknown', 4]]).transform([['divorced', 1], ['married', 2], ['single', 3], ['unknown', 4]])
    data['education'] = enc.fit([['basic.4y', 1], ['basic.6y', 2], ['basic.9y', 3], ['high.school', 4],
                                  ['illiterate', 5], ['professional.course', 6], ['university.degree', 7],
                                  ['unknown', 8]]).transform([['basic.4y', 1], ['basic.6y', 2], ['basic.9y', 3], ['high.school', 4],
                                  ['illiterate', 5], ['professional.course', 6], ['university.degree', 7],
                                  ['unknown', 8]])
    data['default'] = enc.fit([['no', 1], ['yes', 2], ['unknown', 3]]).transform([['no', 1], ['yes', 2], ['unknown', 3]])
    data['housing'] = enc.fit([['no', 1], ['yes', 2], ['unknown', 3]]).transform([['no', 1], ['yes', 2], ['unknown', 3]])
    data['loan'] = enc.fit([['no', 1], ['yes', 2], ['unknown', 3]]).transform([['no', 1], ['yes', 2], ['unknown', 3]])
    data = data.fillna(-999)
    return data

X_train=one_hot_encoder(X_train_0)
X_test=one_hot_encoder(X_test_0)


start_2 = time.time()

""" ACTUALLY TRAINING MY CLASSIFICATION TREE! """
clf_pruned = DecisionTreeClassifier(criterion = "gini", random_state = 20, max_depth=7)
# 100, 20, 10 all do not work when plotting :')
# max_depth=3, min_samples_leaf=5
clf_pruned.fit(X_train, y_train)


# visualizing my tree 
xvar = df_raw.drop('next.product', axis=1)
feature_cols = xvar.columns
dot_data = StringIO()
export_graphviz(clf_pruned, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=uniqueProduct)

(graph, ) = graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree_3.png")
# Image(graph.create_png())
end = time.time()
print("This chunk ran for ", end - start, " seconds!") 
# print("This chunk ran for ", end - start_2, " seconds!") 

""" scoring """
preds_pruned = clf_pruned.predict(X_test)
preds_pruned_train = clf_pruned.predict(X_train)
print(accuracy_score(y_test,preds_pruned))
print(accuracy_score(y_train,preds_pruned_train))

# ======== dummy ======== 
# 3 --> 80.0%
# 5 --> 85.0% 
# 6 --> 85.9% accuracy
# 7 --> 86.3% accuracy
# 8 --> 86.4% accuracy
# 9 --> 86.3% accuracy
# 10 --> 86.0% accuracy
# 12 --> 85.5% accuracy
# 20 --> 82.4% accuracy

# ======== WRONG! label encoding ======== 
# 3 --> 81.3%
# 4 --> 83.8%
# 5 --> 88.3%
# 6 --> 91.9%
# 7 --> 94.4%
# 8 --> 95.3%
# 9 --> 95.8%
# 10 -> 96.4%
# 12 -> 96.8%
# 20 -> 96.6%


# creating a confusion matrix 
cm = confusion_matrix(y_test, preds_pruned)


## Calculating feature importance
feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head(7) 
