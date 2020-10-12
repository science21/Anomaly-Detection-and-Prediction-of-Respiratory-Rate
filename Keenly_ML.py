# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:47:38 2020

@author: Jianyong
"""

import os
path="D:\Dropbox\OMSA\ISYEPractice\Keenly\Data"
os.chdir(path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn import  naive_bayes, linear_model, neighbors, tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


p1_27h=pd.read_pickle('p1_27h')
p1_27v=pd.read_pickle('p1_27v')

pv=p1_27v.loc[:,['Year', 'Month', 'Day', 'Hour','RespirationRate','PID']]
pv=pv.rename(columns={'RespirationRate':'RespirationRateVar'})

ph=p1_27h.reset_index()
pvh=pv.merge(ph, on=['Year', 'Month', 'Day', 'Hour','PID'])

dfkm1=pvh.loc[:,['RespirationRate','RespirationRateVar','Hour', 'PID', 'Distance', 'SignalQuality']].fillna(0).values

##########################################################################
########### K-means clustering ###########################################
from sklearn.cluster import KMeans

wcss =[]
for i in range (1,16):
    kmeans=KMeans(n_clusters=i,  max_iter=100, n_init =10, random_state=42)
    kmeans.fit(dfkm1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,16), wcss)
plt.title('Selecting k clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans1=KMeans(n_clusters=10,  max_iter=100, n_init =10, random_state=42)
y_km1 =kmeans1.fit_predict(dfkm1)

plt.scatter(dfkm1[y_km1==0, 0],dfkm1[y_km1==0, 1], s=100, c='red')
plt.scatter(dfkm1[y_km1==1, 0],dfkm1[y_km1==1, 1], s=100, c='blue')
plt.scatter(dfkm1[y_km1==2, 0],dfkm1[y_km1==2, 1], s=100, c='green')
plt.scatter(dfkm1[y_km1==3, 0],dfkm1[y_km1==3, 1], s=100, c='cyan')
plt.scatter(dfkm1[y_km1==4, 0],dfkm1[y_km1==4, 1], s=100, c='orange')
plt.scatter(dfkm1[y_km1==5, 0],dfkm1[y_km1==5, 1], s=100, c='magenta', label='Anormaly') #Anormaly
plt.scatter(dfkm1[y_km1==6, 0],dfkm1[y_km1==6, 1], s=100, c='yellow')
plt.scatter(dfkm1[y_km1==7, 0],dfkm1[y_km1==7, 1], s=100, c='black')
plt.scatter(dfkm1[y_km1==8, 0],dfkm1[y_km1==8, 1], s=100, c='pink')
plt.scatter(dfkm1[y_km1==9, 0],dfkm1[y_km1==9, 1], s=100, c='red')
plt.title('K-means clusters')
plt.xlabel('Mean respiration rate')
plt.ylabel('Variance of respiration rate')
plt.legend()
plt.show()

# designate anormaly
y_km1[y_km1!=5]=0
y_km1[y_km1==5]=1

pvh['AnormalyKmean']=y_km1


###################################################################
############################ Predicting anormaly ##################


## Using Anormaly detected by Seasona -Trend decomposition as label

data =pvh.loc[:,['Hour', 'RespirationRateVar', 'PID', 'Distance', 'SignalQuality', 'RespirationRate', 
       'AnormalyRes', 'AnormalyKmean']].fillna(0).values #convert data from dataframe to numpy array

xdata = data[:,0:6]
ydata = data[:,6]

#Split the data into training and test data by 70:30
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata, test_size =0.3, random_state=48)
xtrain = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest =scaler.transform(x_test)


## Naive Bayes
NB_model = naive_bayes.GaussianNB()
NB_model.fit(xtrain, y_train)
NB_pred = NB_model.predict(xtest)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_AUC=roc_auc_score(y_test,NB_pred)
NB_predp = NB_model.predict_proba(xtest)
NB_predp=NB_predp[:, 1]
NB_AUC1=roc_auc_score(y_test,NB_predp)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=', NB_AUC1)
NB_fpr1, NB_tpr1, _ = roc_curve(y_test,NB_predp)



## Linear Discriminant Analysis
LDA_model =LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', n_components=1)
LDA_model.fit(xtrain, y_train)
LDA_pred = LDA_model.predict(xtest)
LDA_accuracy= sum(LDA_pred==y_test)/len(y_test)
LDA_AUC=roc_auc_score(y_test,LDA_pred)

LDA_predp = LDA_model.predict_proba(xtest)
LDA_predp=LDA_predp[:, 1]
LDA_AUC1=roc_auc_score(y_test,LDA_predp)
LDA_fpr1, LDA_tpr1, _ = roc_curve(y_test,LDA_predp)
print('Linear Discriminant Analysis: ', 'Accuracy=', LDA_accuracy, 'AUC=', LDA_AUC1)


## Logistic regression
LR_model = linear_model.LogisticRegression(solver='lbfgs')
LR_model.fit(xtrain, y_train)
LR_pred = LR_model.predict(xtest)
LR_accuracy= sum(LR_pred==y_test)/len(y_test)
LR_AUC=roc_auc_score(y_test,LR_pred)

LR_predp = LR_model.predict_proba(xtest)
LR_predp=LR_predp[:, 1]
LR_AUC1=roc_auc_score(y_test,LR_predp)
LR_fpr1, LR_tpr1, _ = roc_curve(y_test,LR_predp)
print('Logistic Regression: ', 'Accuracy=', LR_accuracy, 'AUC=', LR_AUC1)



## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(xtrain, y_train)
KNN_pred = KNN_model.predict(xtest)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_AUC=roc_auc_score(y_test,KNN_pred)

KNN_predp = KNN_model.predict_proba(xtest)
KNN_predp=KNN_predp[:, 1]
KNN_AUC1=roc_auc_score(y_test,KNN_predp)
KNN_fpr1, KNN_tpr1, _ = roc_curve(y_test,KNN_predp)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=', KNN_AUC1)


#  SVC method
SVC_model= SVC(gamma='auto', C=1, max_iter=2000,probability=True,random_state=90, tol=1e-4)
SVC_model.fit(xtrain, y_train)
SVC_pred =SVC_model.predict(xtest)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_AUC=roc_auc_score(y_test,SVC_pred)

SVC_predp = SVC_model.predict_proba(xtest)
SVC_predp=SVC_predp[:, 1]
SVC_AUC1=roc_auc_score(y_test,SVC_predp)
SVC_fpr1, SVC_tpr1, _ = roc_curve(y_test,SVC_predp)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=', SVC_AUC1)

## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 3),random_state=48)
NN_model.fit(xtrain, y_train)
NN_pred = NN_model.predict(xtest)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_AUC=roc_auc_score(y_test,NN_pred)

NN_predp = NN_model.predict_proba(xtest)
NN_predp=NN_predp[:, 1]
NN_AUC1=roc_auc_score(y_test,NN_predp)
NN_fpr1, NN_tpr1, _ = roc_curve(y_test,NN_predp)
print('Neural Network: ', 'Accuracy=', NN_accuracy, 'AUC=', NN_AUC1)

## Decision tree

DT_model =tree.DecisionTreeClassifier(max_depth = 3, random_state =48)
DT_model.fit(xtrain, y_train)
DT_pred =DT_model.predict(xtest)
DT_accuracy=sum(DT_pred==y_test)/len(y_test)
DT_AUC=roc_auc_score(y_test,DT_pred)

DT_predp = DT_model.predict_proba(xtest)
DT_predp=DT_predp[:, 1]
DT_AUC1=roc_auc_score(y_test,DT_predp)
DT_fpr1, DT_tpr1, _ = roc_curve(y_test,DT_predp)
print('Decision Tree: ', 'Accuracy=', DT_accuracy, 'AUC=', DT_AUC1)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 15, random_state =90)
RF_model.fit(xtrain, y_train)
RF_pred =RF_model.predict(xtest)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_AUC=roc_auc_score(y_test,RF_pred)

RF_predp = RF_model.predict_proba(xtest)
RF_predp=RF_predp[:, 1]
RF_AUC1=roc_auc_score(y_test,RF_predp)
RF_fpr1, RF_tpr1, _ = roc_curve(y_test,RF_predp)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=', RF_AUC1)
RF_model.feature_importances_
## Adaboost
AB_model =AdaBoostClassifier(n_estimators=15, random_state=90)
AB_model.fit(xtrain, y_train)
AB_pred =AB_model.predict(xtest)
AB_accuracy=sum(AB_pred==y_test)/len(y_test)
AB_AUC=roc_auc_score(y_test,AB_pred)

AB_predp = AB_model.predict_proba(xtest)
AB_predp=AB_predp[:, 1]
AB_AUC1=roc_auc_score(y_test,AB_predp)
AB_fpr1, AB_tpr1, _ = roc_curve(y_test,AB_predp)
print('Adaboost: ', 'Accuracy=', AB_accuracy, 'AUC=', AB_AUC1)


####################################################################
## Using Anormaly detected by Seasona -Trend decomposition as label

ydata2 = data[:,7]

#Split the data into training and test data by 75:25
x_train, x_test, y_train, y_test =model_selection.train_test_split(xdata, ydata2, test_size =0.3, random_state=48)
xtrain = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
xtest =scaler.transform(x_test)


## Naive Bayes
NB_model = naive_bayes.GaussianNB()
NB_model.fit(xtrain, y_train)
NB_pred = NB_model.predict(xtest)
NB_accuracy= sum(NB_pred==y_test)/len(y_test)
NB_AUC=roc_auc_score(y_test,NB_pred)
NB_predp = NB_model.predict_proba(xtest)
NB_predp=NB_predp[:, 1]
NB_AUC2=roc_auc_score(y_test,NB_predp)
print('Naive Bayes: ', 'Accuracy=', NB_accuracy, 'AUC=', NB_AUC2)
NB_fpr2, NB_tpr2, _ = roc_curve(y_test,NB_predp)



## Linear Discriminant Analysis
LDA_model =LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', n_components=1)
LDA_model.fit(xtrain, y_train)
LDA_pred = LDA_model.predict(xtest)
LDA_accuracy= sum(LDA_pred==y_test)/len(y_test)
LDA_AUC=roc_auc_score(y_test,LDA_pred)

LDA_predp = LDA_model.predict_proba(xtest)
LDA_predp=LDA_predp[:, 1]
LDA_AUC2=roc_auc_score(y_test,LDA_predp)
LDA_fpr2, LDA_tpr2, _ = roc_curve(y_test,LDA_predp)
print('Linear Discriminant Analysis: ', 'Accuracy=', LDA_accuracy, 'AUC=', LDA_AUC2)


## Logistic regression
LR_model = linear_model.LogisticRegression(solver='lbfgs')
LR_model.fit(xtrain, y_train)
LR_pred = LR_model.predict(xtest)
LR_accuracy= sum(LR_pred==y_test)/len(y_test)
LR_AUC=roc_auc_score(y_test,LR_pred)

LR_predp = LR_model.predict_proba(xtest)
LR_predp=LR_predp[:, 1]
LR_AUC2=roc_auc_score(y_test,LR_predp)
LR_fpr2, LR_tpr2, _ = roc_curve(y_test,LR_predp)
print('Logistic Regression: ', 'Accuracy=', LR_accuracy, 'AUC=', LR_AUC2)



## KNN
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(xtrain, y_train)
KNN_pred = KNN_model.predict(xtest)
KNN_accuracy= sum(KNN_pred==y_test)/len(y_test)
KNN_AUC=roc_auc_score(y_test,KNN_pred)

KNN_predp = KNN_model.predict_proba(xtest)
KNN_predp=KNN_predp[:, 1]
KNN_AUC2=roc_auc_score(y_test,KNN_predp)
KNN_fpr2, KNN_tpr2, _ = roc_curve(y_test,KNN_predp)
print('KNN: ', 'Accuracy=', KNN_accuracy, 'AUC=', KNN_AUC2)


#  SVC method
SVC_model= SVC(gamma='auto', C=1, max_iter=2000,probability=True,random_state=90, tol=1e-4)
SVC_model.fit(xtrain, y_train)
SVC_pred =SVC_model.predict(xtest)
SVC_accuracy= sum(SVC_pred==y_test)/len(y_test)
SVC_AUC=roc_auc_score(y_test,SVC_pred)

SVC_predp = SVC_model.predict_proba(xtest)
SVC_predp=SVC_predp[:, 1]
SVC_AUC2=roc_auc_score(y_test,SVC_predp)
SVC_fpr2, SVC_tpr2, _ = roc_curve(y_test,SVC_predp)
print('Support Vector Machine: ', 'Accuracy=', SVC_accuracy, 'AUC=',SVC_AUC2)

## Neural network
NN_model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 3),random_state=48)
NN_model.fit(xtrain, y_train)
NN_pred = NN_model.predict(xtest)
NN_accuracy= sum(NN_pred==y_test)/len(y_test)
NN_AUC=roc_auc_score(y_test,NN_pred)

NN_predp = NN_model.predict_proba(xtest)
NN_predp=NN_predp[:, 1]
NN_AUC2=roc_auc_score(y_test,NN_predp)
NN_fpr2, NN_tpr2, _ = roc_curve(y_test,NN_predp)
print('Neural Network: ', 'Accuracy=', NN_accuracy, 'AUC=', NN_AUC2)


## Decision tree
DT_model =tree.DecisionTreeClassifier(max_depth = 3, random_state =48)
DT_model.fit(xtrain, y_train)
DT_pred =DT_model.predict(xtest)
DT_accuracy=sum(DT_pred==y_test)/len(y_test)
DT_AUC=roc_auc_score(y_test,DT_pred)

DT_predp = DT_model.predict_proba(xtest)
DT_predp=DT_predp[:, 1]
DT_AUC2=roc_auc_score(y_test,DT_predp)
DT_fpr2, DT_tpr2, _ = roc_curve(y_test,DT_predp)
print('Decision Tree: ', 'Accuracy=', DT_accuracy, 'AUC=', DT_AUC2)


## Random Forest
RF_model =RandomForestClassifier(n_estimators = 15, random_state =90)
RF_model.fit(xtrain, y_train)
RF_pred =RF_model.predict(xtest)
RF_accuracy=sum(RF_pred==y_test)/len(y_test)
RF_AUC=roc_auc_score(y_test,RF_pred)

RF_predp = RF_model.predict_proba(xtest)
RF_predp=RF_predp[:, 1]
RF_AUC2=roc_auc_score(y_test,RF_predp)
RF_fpr2, RF_tpr2, _ = roc_curve(y_test,RF_predp)
print('Random Forest: ', 'Accuracy=', RF_accuracy, 'AUC=', RF_AUC2)


## Adaboost
AB_model =AdaBoostClassifier(n_estimators=15, random_state=90)
AB_model.fit(xtrain, y_train)
AB_pred =AB_model.predict(xtest)
AB_accuracy=sum(AB_pred==y_test)/len(y_test)
AB_AUC=roc_auc_score(y_test,AB_pred)

AB_predp = AB_model.predict_proba(xtest)
AB_predp=AB_predp[:, 1]
AB_AUC2=roc_auc_score(y_test,AB_predp)
AB_fpr2, AB_tpr2, _ = roc_curve(y_test,AB_predp)
print('Adaboost: ', 'Accuracy=', AB_accuracy, 'AUC=', AB_AUC2)


#######
##################### PLot AUC of ROC  #############


fig, axs = plt.subplots(3, 3)
axs[0, 0].plot(NB_fpr1, NB_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % NB_AUC1)
axs[0, 0].plot(NB_fpr2, NB_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % NB_AUC2)
axs[0, 0].set(ylabel='True Positive Rate') 
axs[0, 0].set_title('Naive Bayes') 
axs[0, 0].legend(loc = 'lower right')

axs[0, 1].plot(LDA_fpr1, LDA_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % LDA_AUC1)
axs[0, 1].plot(LDA_fpr2, LDA_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % LDA_AUC2)
axs[0, 1].set_title('Linear Discriminant Analysis') 
axs[0, 1].legend(loc = 'lower right')

axs[0, 2].plot(LR_fpr1, LR_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % LR_AUC1)
axs[0, 2].plot(LR_fpr2, LR_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % LR_AUC2)
axs[0, 2].set_title('Logistic regression') 
axs[0, 2].legend(loc = 'lower right')

axs[1, 0].plot(KNN_fpr1, KNN_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % KNN_AUC1)
axs[1, 0].plot(KNN_fpr2, KNN_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % KNN_AUC2)
axs[1, 0].set_title('K-Nearest Neighbors') 
axs[1, 0].set(ylabel='True Positive Rate') 
axs[1, 0].legend(loc = 'lower right')

axs[1, 1].plot(SVC_fpr1, SVC_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % SVC_AUC1)
axs[1, 1].plot(SVC_fpr2, SVC_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % SVC_AUC2)
axs[1, 1].set_title('Support Vector Machine') 
axs[1, 1].legend(loc = 'lower right')

axs[1, 2].plot(NN_fpr1, NN_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % NN_AUC1)
axs[1, 2].plot(NN_fpr2, NN_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % NN_AUC2)
axs[1, 2].set_title('Neural Network') 
axs[1, 2].legend(loc = 'lower right')

axs[2, 0].plot(DT_fpr1, DT_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % DT_AUC1)
axs[2, 0].plot(DT_fpr2, DT_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % DT_AUC2)
axs[2, 0].set_title('Decision Tree') 
axs[2, 0].set(xlabel='False Positive Rate', ylabel='True Positive Rate') 
axs[2, 0].legend(loc = 'lower right')


axs[2, 1].plot(RF_fpr1, RF_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % RF_AUC1)
axs[2, 1].plot(RF_fpr2, RF_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % RF_AUC2)
axs[2, 1].set_title('Random Forest') 
axs[2, 1].set(xlabel='False Positive Rate') 
axs[2, 1].legend(loc = 'lower right')


axs[2, 2].plot(AB_fpr1, AB_tpr1, marker='.', label = 'Group 1 (AUC = %0.2f)' % AB_AUC1)
axs[2, 2].plot(AB_fpr2, AB_tpr2, marker='.', label = 'Group 2 (AUC = %0.2f)' % AB_AUC2)
axs[2, 2].set_title('Adaboost') 
axs[2, 2].set(xlabel='False Positive Rate') 
axs[2, 2].legend(loc = 'lower right')













