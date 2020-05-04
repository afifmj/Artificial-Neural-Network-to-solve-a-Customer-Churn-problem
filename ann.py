#--------------------------------Data Preprocessing----------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#First we encode the countries
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Then we encode the genders
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#Creating dummy columns for the countries
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#----------------------------End of data preprocessing----------------------------------

#-----------------------------Building the ANN------------------------------------------------------

import keras
from keras.models import Sequential #To initalize our neural network
from keras.layers import Dense #To create the layers of the ANN
from keras.layers import Dropout

classifier = Sequential() #initalizing the ann

#adding the input layer and first hidden layer - the hidden layer has 6 nodes, input layer has 11 nodes
#The activation func is rectifier func. and the kernel_initalizer sets the weights near to 0
classifier.add(Dense(units=6, kernel_initializer= 'uniform', activation= 'relu', input_dim = 11 ))
classifier.add(Dropout(rate=0.1))#Applying dropout to first hidden layer to prevent overfitting

#adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer= 'uniform', activation= 'relu'))
classifier.add(Dropout(rate=0.1))#Applying dropout to second hidden layer to prevent overfitting

#adding the output layer with one node and sigmoid activation func.
classifier.add(Dense(units=1, kernel_initializer= 'uniform', activation= 'sigmoid'))


#Compiling the ann. We use the stochastic gradient descent algo called 'adam'. the loss func which we will be
#using is called 'binary_crossentropy'.
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#training the ann. We use a batch size of 10 (the no. of rows it evaluates before updating the weights)
#  ,and we use 100 epochs
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

#---------------------------------------------------------------------------------------------------


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print(new_prediction)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#--------------------Evaluating the ANN---------------------------------------------------------------

# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
# mean = accuracies.mean()
# variance = accuracies.std()

# print(accuracies)
# print(mean)
# print(variance)
#-------------------------------------------------------------------------------------------------------

#------------------------------------Tuning the ANN-------------------------------------------------------
# # Tuning the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn = build_classifier)
# parameters = {'batch_size': [25, 32],
#               'epochs': [100, 500],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train, y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
#-----------------------------------------------------------------------------------------------------------------
