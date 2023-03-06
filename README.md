# Random-forest

This is a fully functional example of a random forest project created to make predictions on data with good accuracy.

Here we are going to use sklearn to model our random forest.
There are steps we will be taking to create our random forest which are;
-Checking for any null data
-Splitting and Training data
-Fitting the model
-Assessing the performance
-Visualizing our model

The packages and functions used here are;

# For Data Processing
*import pandas as pd
*import numpy as np

#To split and train data sets we use
*x_train,x_test,y_train,y_test

# For Modelling
*from sklearn.ensemble import RandomForestClassifier
*from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
*from sklearn.model_selection import RandomizedSearchCV, train_test_split

#Because the random forest I was working on was not in array form, I used the One hot encoder to convert the data into numerical form.
*from sklearn.preprocessing import OneHotEncoder

#After making the conversion, we can now proceed to fit our trained model using;
rf.fit(x_train, y_train)

We can proceed to check the accuracy score of our model. I got an accuracy score of 50, which is not the most accurate. I will make adjustments since this is my first attempt at creating this model.
#To check accuracy score
*sklearn.metrics import accuracy_score
*accuracy_score(y_true, y_pred)


The below code plots the importance of each feature, using the model’s internal score to find the best way to split the data within each decision tree.
#Visualization
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

Since the data was changed to numerical values, the variables are displayed in numbers.
