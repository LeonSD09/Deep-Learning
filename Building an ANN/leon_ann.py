# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Update everything 
# conda update --all

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Make an ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# One way to choose the number of nodes in hidden layer:
# Average length of the input and output layer
# In this case our input layer has 11 features, output is a single column
# Therefore, 11+1 / 2 = 6 .. output_dim = 6
# input_dim = number of features = 11
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# Adding another hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
# (performing stochastic gradient descent)
# 'adam' is a stochastic gradient descent algorithm
# binary_crossentropy loss used when binary outcome (like in this case)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) # returns the probabilities 
y_pred = (y_pred > 0.5) # returns True/False (1/0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Homework

"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?
"""

# Need to format the new input like original training feature set
# One hot encoding
# Standard scaling
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
# False

# Part 5 - Evaluating, Improving and Tuning the ANN

# Reimport the keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
# Function that builds the same classifier as built above (without comments)    
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Wrap the classifier function built above
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

# The 10 accuracies returned by k-fold cross validation
# A cv (num of folds) of 10 is recommended most of the time
# n_jobs = -1 will use all of the available compute
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean_accuracies = accuracies.mean()
acc_variance = accuracies.std()

print(mean_accuracies, acc_variance)
# 0.8372499950975181 0.014745761639726728

# Improving the ANN
# Dropout regularization to reduce overfitting if needed

def build_classifier_w_dropout():
# Function that builds the same classifier as built above (without comments)    
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
# Function that builds the same classifier as built above (without comments)    
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Wrap the classifier function built above
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25, 32], 
              'epochs':[100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters, 
                           scoring='accuracy', 
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ # {'batch_size': 32, 'epochs': 500, 'optimizer': 'rmsprop'}
best_accuracy = grid_search.best_score_ # 0.856875

