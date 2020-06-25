#author:Rrutum Lavana

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Nursery.csv')
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, [8]]

X = pd.get_dummies(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
    
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(np.ravel(y))

from sklearn.ensemble import ExtraTreesClassifier
# Feature Extraction
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)
print(model.feature_importances_)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Feature Extraction
test = SelectKBest(score_func=chi2, k=6)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

X_new = SelectKBest(chi2, k=6).fit_transform(X, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=18)

def support_vector_machine(X_train,X_test,y_train,y_test):
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf',gamma='scale')

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("SVM:")
    accuracy(y_test,y_pred)

def random_forest(X_train, X_test, y_train, y_test):
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    print("Random Forest:")
    accuracy(y_test,y_pred)

def logistic_regression(X_train, X_test, y_train, y_test):
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0,solver='liblinear',multi_class='auto')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Logistic Regression:")
    accuracy(y_test,y_pred)

def xgboost(X_train, X_test, y_train, y_test):    
    # Fitting XGBoost to the Training set
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    print("XGBoost:")
    accuracy(y_test,y_pred)

def accuracy(y_test,y_pred):
    #Making the Confusion Matrix
    #from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    print("Accuracy=",accuracy_score(y_test, y_pred)*100)

# Applying classification algorithms and comparing their accuracies
support_vector_machine(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)
xgboost(X_train, X_test, y_train, y_test)
