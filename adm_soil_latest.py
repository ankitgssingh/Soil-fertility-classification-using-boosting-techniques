# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:09:34 2019

@author: Ankit
"""

#IMPORTING ALL THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

#the file soil_data is also attached in the submitted zip as excel has been used to clean the source file first.
dataset = pd.read_csv("soil_data.csv")   #reading the csv
'''dataset.info()        #checking the column names and row names
dataset.shape            #checking the number of rows and columns
dataset.describe()'''    #checking the description of rows and columns

'''removing any anomalies from column names'''
dataset.columns =dataset.columns.str.lower()   #converting all the columns names to lower cases
dataset.columns =dataset.columns.str.strip()   #removing any gap between column names
dataset.columns =dataset.columns.str.replace(' ', '_')   #replacing all spaces within column names with ’_’


'''taking list of categorical and numeric column name'''
object_cols = [ ]    #making empty list to store categorical/string columns
non_object_cols = [ ]   #making empty list to store numerical columns
object_cols = list(dataset.select_dtypes(include=['object']).columns)     #moving the categorical/string column names into the list
non_object_cols = list(dataset.select_dtypes(exclude=['object']).columns)    #moving the numerical column names into the list


'''removing duplicate but didnt find any duplicate row'''    
dataset = dataset.drop_duplicates(subset = None,keep = 'first')
'''dataset.nunique()'''    #checking how many unique values


'''Filling missing values in numerical values'''
imputer = Imputer(missing_values = 'NaN',strategy= 'median',axis = 0)
imputer = imputer.fit(dataset[non_object_cols])
dataset[non_object_cols] = imputer.transform(dataset[non_object_cols])


'''Filling missing values in categorical values'''
for c in object_cols:
    dataset[c].fillna(dataset[c].value_counts().index[0],inplace = True)
'''dataset.info()'''

''' Checking correlation'''
corr = dataset.corr()
ax = sns.heatmap(corr, annot=True)

'''dropping the column as it has high correlation with other independent variable'''
dataset =  dataset.drop(['macroporosity-m10'], axis = 1)
'''taking list of categorical and numeric column names again as we have removed one column due to multicollinearity'''
object_cols = list(dataset.select_dtypes(include=['object']).columns)
non_object_cols = list(dataset.select_dtypes(exclude=['object']).columns)


'''ENCODING'''
'''label encoding for dependent variable'''
target_column = 'landuse'
label_encoder_class = preprocessing.LabelEncoder()#encoding the target variables
label_encoder = label_encoder_class.fit_transform(dataset[target_column])

'''dummy encoding for all categorical variables'''
one_hot_cols = ['region','soilorder','meets_ph','meets_bulkdensity','meets_macroporosity','meets_olsenp', 'meets_totalc','meets_totaln','meets_amn','meets_organicr','meets_physicals']
object_cols_values = dataset.select_dtypes(include=['object'])
object_cols_values = object_cols_values.drop('landuse',axis = 1)

'''Get dummy encoding'''
empty_df = pd.DataFrame()
for c in one_hot_cols:
    dfDummies = pd.get_dummies(object_cols_values[c],prefix = 'category')
    ''' Dropping extra column B as it is now encoded'''
    dfDummies=dfDummies.iloc[:,1:]
    empty_df = pd.concat([empty_df, dfDummies], axis=1)


'''creating dummy values'''
empty_df = pd.get_dummies(object_cols_values, prefix=['region','soilorder','meets_ph','meets_bulkdensity', 'meets_macroporosity','meets_olsenp','meets_totalc','meets_totaln','meets_amn','meets_organicr','meets_physicals'])
dataset.drop(labels = one_hot_cols,axis = 1, inplace = True)
dataset['landuse'] = label_encoder
final_dataset = pd.concat([empty_df,dataset], axis=1) #concatenation of all the categorical variables with numerical ones and dependent
'''saving the output file'''
'''final_dataset.to_csv('/Users/Ankit/Documents/final_soil_data.csv')'''

#to find the indepent variable with the most relation with dependent landuse
corr3 = dataset.corr()
ax3 = sns.heatmap(corr3, annot=True)
#PH is the most contributing variable to landuse

'''final_dataset = pd.read_csv("final_soil_data.csv") '''
########################################################
'''PREPARING FOR Models'''
x = final_dataset.iloc[:, 0:54]   #storing all the columns except the dependent column
y = final_dataset.iloc[:, -1]    #storing the dependent column


'''train,test,split'''  # Training the model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 123) #setting a seed value
'''final_dataset["landuse"].value_counts()'''

######################################
# Feature Scaling to get better performance validation and results
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
X_train = sc1.fit_transform(x_train)
X_test = sc1.transform(x_test)


########################################
from sklearn.metrics import confusion_matrix, accuracy_score   #cofusion matrix
from sklearn import svm
svm_model = svm.SVC(kernel='linear', C=1, gamma='auto')
svm_model.fit(X_train,y_train)
predictions_svm = svm_model.predict(X_test)
accuracy_svm=accuracy_score(predictions_svm, y_test)
cm_svm = confusion_matrix(y_test, predictions_svm)
tp, fp, fn, tn = confusion_matrix(y_test, predictions_svm).ravel()
specificity_svm = tn / (tn+fp)  
specificity_svm
print("specivity of svm : %.2f%%" % (specificity_svm * 100.0))
sensivity_svm = tp / (tp+fn)
print("sensivity of svm : %.2f%%" % (sensivity_svm * 100.0))
print("Accuracy of svm : %.2f%%" % (accuracy_svm * 100.0))
'''Accuracy of svm : 89.66%'''
'''sensivity of svm : 90.77%'''
'''specificity is 86%'''
kappa_svm = cohen_kappa_score(y_test, predictions_svm)
print(kappa_svm)
'''kappa is .7'''


'''10 fold cross validation'''
from sklearn.model_selection import cross_val_score
scores_svm = cross_val_score(svm_model, X_train, y_train, cv=10)
print("Validated Accuracy of svm : %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))
''' Validated Accuracy of svm: 0.94 (+/- 0.05)'''
#good performance with low standard deviation 

#grid search was used to find the best parameters for svm, default parameters were the best
'''from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 100, 1000], 'kernel' : ['rbf']},
              {'C': [1, 100, 1000], 'kernel' : ['linear']}]
grid_search = GridSearchCV(estimator = svm_model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_'''

##########################################################
#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)    #predicting
#building the confusion matrix
cm = confusion_matrix(y_test,y_pred)   #Comparing the test result with the predicte result of the model
accuracy = accuracy_score(y_test,y_pred)   #getting the accuracy of the model
tp1, fp1, fn1, tn1 = confusion_matrix(y_test, y_pred).ravel()
specificity_lr = tn1 / (tn1+fp1)  #
print("specificity of logistic reg : %.2f%%" % (specificity_lr * 100.0))
print("Accuracy of logistic reg : %.2f%%" % (accuracy * 100.0))
sensivity_lr = tp1 / (tp1+fn1)
kappa_lr = cohen_kappa_score(y_test, y_pred)
print(kappa_lr)
print( np.unique(y_pred)) #checking i the model is predicting both values
print("sensitivy of logistic reg : %.2f%%" % (sensivity_lr * 100.0))
'''sensitivy is 93.8'''
''' Accuracy of logistic reg : 94.25% '''
'''kappa is .85. better than svm'''
'''specificity is 95.4%'''


'''10 fold cross validation'''
from sklearn.model_selection import cross_val_score
scores_lr = cross_val_score(classifier, X_train, y_train, cv=10)
print("Validated Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))
''' Validated Accuracy: 0.95 (+/- 0.06)'''
#accuracy is very high with low standard deviation, very good performance 

################################################

'''xgboost'''
#pip install xgboost
from xgboost import XGBClassifier
#from xgboost.xgbclassifier import XGBClassifier
# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data
y_pred1 = model.predict(x_test)
predictions = [round(value) for value in y_pred1]
# evaluate predictions
cm1 = confusion_matrix(y_test, y_pred1)
accuracy_xgboost = accuracy_score(y_test, predictions)
tp2, fp2, fn2, tn2 = confusion_matrix(y_test, predictions).ravel()
specificity_xgb = tn2 / (tn2+fp2) 
sensivity_xg = tp2 / (tp2+fn2)
print("sensitiy of xg boost: %.2f%%" % (sensivity_xg * 100.0))
print("specificity of xg boost: %.2f%%" % (specificity_xgb * 100.0))
print("Accuracy of xg boost: %.2f%%" % (accuracy_xgboost * 100.0))
kappa_xg = cohen_kappa_score(y_test, predictions)
print(kappa_xg)
'''kappa for xgboost is .88, better than svm and logistic regression'''
'''specificity is 95.6%'''
'''sensitiy is 95.3'''
'''Accuracy of xg boost: 95.40%'''


'''10 fold cross validation for training'''
from sklearn.model_selection import cross_val_score
scores_xgboost = cross_val_score(model, X_train, y_train, cv=10)
print("Validated Accuracy: %0.2f (+/- %0.2f)" % (scores_xgboost.mean(), scores_xgboost.std() * 2))
''' Validated Accuracy: 0.92 (+/- 0.09)'''
#very good accuracy with low standard deviation

'''Grid search to find the best parameters'''
parameters3 = [{'learning_rate' : [.001, .001, .01, .02], 'n_estimators' : [50, 100, 200, 500]},
              {'learning_rate' : [.001, .001, .01, .02], 'n_estimators' : [1000]}]
grid_search2 = GridSearchCV(estimator = model,
                           param_grid = parameters3,
                           scoring = 'accuracy',
                           cv = 10)
grid_search2 = grid_search2.fit(X_train, y_train)
best_accuracy4 = grid_search2.best_score_
best = grid_search2.best_params_
# we already got the best accuracy by default parameters 




######################################
'''catboost'''
#pip install catboost
from catboost import CatBoostClassifier
#fitting model
cat = CatBoostClassifier(iterations=500, learning_rate=.02)
cat.fit(x_train, y_train)  #it will take 30 secs to learn
#prediction
y_pred2 = cat.predict(x_test)
#accuracy of catboost
cm2 = confusion_matrix(y_test, y_pred2)
accuracy_catboost = accuracy_score(y_test, y_pred2)
tp3, fp3, fn3, tn3 = confusion_matrix(y_test, y_pred2).ravel()
specificity_cat = tn3 / (tn3+fp3) 
sensivity_cat = tp3 / (tp3+fn3)
print("specificty of catboost : %.2f%%" % (specificity_cat * 100.0))  #best accurcy
print("Accuracy of catboost : %.2f%%" % (accuracy_catboost * 100.0))  #best accurcy 
print("sensitivity of catboost : %.2f%%" % (sensivity_cat * 100.0))  #best accurcy 

kappa_cat = cohen_kappa_score(y_test, y_pred2)
print(kappa_cat) 
'''sensitivity is 98.3 is the highest''' 
'''specificity is 95.8 is the highest'''
'''Accuracy of catboost : 97.70% is the highest''' 
'''kappa score is .94 which is the highest'''

print( np.unique(y_pred2)) #checking if the model is predicting both values

'''cross validation'''
from sklearn.model_selection import cross_val_score
scores_catboost = cross_val_score(cat, X_train, y_train, cv=10) #it will take a minute to fit
print("Validated Accuracy: %0.2f (+/- %0.2f)" % (scores_catboost.mean(), scores_catboost.std() * 2))
''' Validated Accuracy: 0.94 (+/- 0.07)'''
#A very good accuracy with low deviation 
'''catboost got the best accuracy, best kappa score and best sensivity'''

'''grid search
from sklearn.model_selection import GridSearchCV
parameters1 = [{'iterations': [100, 200, 500, 700, 1000], 'learning_rate' : [.0001, .001, .01, .02]},
              {'iterations' : [100, 200, 500, 700, 1000], 'learning_rate' : [.03]}]
grid_search1 = GridSearchCV(estimator = cat,
                           param_grid = parameters1,
                           scoring = 'accuracy',
                           cv = 10)
grid_search1 = grid_search1.fit(X_train, y_train)
best_accuracy1 = grid_search.best_score_   '''






