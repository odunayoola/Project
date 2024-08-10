#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing necessary libraries
import os  # provides functions for interacting with the operating system
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# To install sklearn type "pip install numpy scipy scikit-learn" in the anaconda terminal

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

# Datetime library
from pandas import to_datetime
import itertools
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


new_data = pd.read_excel('C:/Users/Shola/Downloads/E Commerce Dataset.xlsx', sheet_name = 'Clean Data')


# In[12]:


new_data.info()


# In[14]:


# Investigate all the elements within each Feature 
for column in new_data:
    unique_vals = np.unique(new_data[column])
    nr_values = len(unique_vals)
    if nr_values < 12:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# In[16]:


custdemo = [ 'CityTier', 'Gender', 'MaritalStatus', 'NumberOfAddress', 'NumberOfDeviceRegistered','WarehouseToHome']
# Feature Lists of Customer Purchasing Behaviour
custbehv = [ 'Tenure', 'PreferredLoginDevice', 'PreferredPaymentMode', 'HourSpendOnApp', 'PreferedOrderCat', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 
            'DaySinceLastOrder', 'CashbackAmount']
# Feature List of Customer Compliance
custcompl = ['SatisfactionScore', 'Complain']


# In[20]:


numerical_data = new_data.select_dtypes(include=['number'])

# Check if numerical_data is not empty
if not numerical_data.empty:
    # Setting the size of the figure for the heatmap
    fig = plt.figure(figsize=(10, 8))

    # Plotting the correlation heatmap using seaborn library
    dataplot = sns.heatmap(numerical_data.corr(), cmap="YlGnBu", annot=True)

    # Adding labels and title
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')

    # Displaying the heatmap
    plt.show()
else:
    print("No numerical data available to plot the heatmap.")


# In[22]:


# Ensure that only numerical columns are used for the correlation matrix
num_data = [ 'Churn', 'Tenure', 'CityTier','WarehouseToHome', 'HourSpendOnApp','NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'Complain',
       'OrderAmountHikeFromlastYear', 'CouponUsed','DaySinceLastOrder', 'CashbackAmount']
numeric_data = new_data[num_data]

# Setting the size of the figure for the heatmap
fig = plt.figure(figsize=(10, 8))

# Plotting the correlation heatmap using seaborn library
dataplot = sns.heatmap(numeric_data.corr(), cmap="YlGnBu", annot=True, fmt=".2f")

# Adding labels and title
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')

# Displaying the heatmap
plt.show()


# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools

# Assuming you have loaded your dataset into a pandas DataFrame `new_raw_data`
# Example:
# new_raw_data = pd.read_csv('your_dataset.csv')

# Separate features and target variable
X = new_data.drop(columns=['Churn'])
y = new_data['Churn']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=5630)

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_prob)

# Calculate Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Print the scores
print(f'ROC AUC: {roc_auc}')
print(f'Balanced Accuracy: {balanced_acc}')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate overall predicted percentage
overall_predicted_percentage = accuracy * 100
print(f'Overall Predicted Percentage: {overall_predicted_percentage:.2f}%')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Generate classification report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert counts to percentage
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'  # Use '.2f' for percentages, '.0f' for counts
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt) + '%',  # Append '%' to the formatted value
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)  # Turn off gridlines

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=[0, 1])  # Assuming binary classification with labels 0 and 1
plt.show()


# In[53]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 500]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best estimator
print("Best parameters found: ", grid_search.best_params_)
print("Best estimator found: ", grid_search.best_estimator_)

# Predict with the best estimator on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}%".format(accuracy * 100))


# In[116]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_auc_score, balanced_accuracy_score

model_grid = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    
model_grid.fit(X_train, y_train) 
y_pred_grid = model_grid.predict(X_test) 
print(classification_report(y_pred_grid, y_test)) 



# In[57]:


from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(LogisticRegression(), 
                                   param_grid) 
random_search.fit(X_train, y_train) 
print(random_search.best_estimator_) 


# In[59]:


model_random = LogisticRegression(C=10, max_iter=500, solver='newton-cg')
model_random.fit(X_train, y_train) 
y_pred_rand = model_random.predict(X_test) 
print(classification_report(y_pred_rand, y_test))


# In[32]:


import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load your data (example)
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target)

# Assuming new_data2 is your DataFrame
# Convert categorical features to numeric
X = pd.get_dummies(new_data.drop(columns=['Churn']))
y = new_data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_prob)

# Calculate Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Print the scores
print(f'ROC AUC: {roc_auc}')
print(f'Balanced Accuracy: {balanced_acc}')

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert counts to percentage
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'  # Use '.2f' for percentages, '.0f' for counts
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt) + '%',  # Append '%' to the formatted value
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)  # Turn off gridlines

rf = RandomForestClassifier(n_estimators=5630, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

train_accuracy = rf.score(X_train, y_train)
test_accuracy = rf.score(X_test, y_test)
# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))

# Calculate overall prediction percentage based on testing accuracy
overall_prediction_percentage = test_accuracy * 100
print(f"Overall Prediction Percentage: {overall_prediction_percentage:.2f}%")

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

# Plotting the confusion matrix
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)
plt.show()


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Assuming param_grid is already defined, for example:
param_grid = { 
	'n_estimators': [25, 50, 100, 150], 
	'max_features': ['sqrt', 'log2', None], 
	'max_depth': [3, 6, 9], 
	'max_leaf_nodes': [3, 6, 9], 
} 

# Create the GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best estimator found by the grid search
print(grid_search.best_estimator_)


# In[39]:


model_grid = RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9)
    
model_grid.fit(X_train, y_train) 
y_pred_grid = model_grid.predict(X_test) 
print(classification_report(y_pred_grid, y_test)) 


# In[41]:


model_grid = RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9, n_estimators=25)
    
model_grid.fit(X_train, y_train) 
y_pred_grid = model_grid.predict(X_test) 
print(classification_report(y_pred_grid, y_test)) 


# In[45]:


from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(RandomForestClassifier(), 
                                   param_grid) 
random_search.fit(X_train, y_train) 
print(random_search.best_estimator_) 


# In[49]:


model_random = RandomForestClassifier(max_depth=9, max_features=None, max_leaf_nodes=9,
                       n_estimators=50) 
model_random.fit(X_train, y_train) 
y_pred_rand = model_random.predict(X_test) 
print(classification_report(y_pred_rand, y_test)) 


# In[89]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
new_data = pd.read_excel('C:/Users/Shola/Downloads/E Commerce Dataset.xlsx', sheet_name = 'Clean Data')

X = pd.get_dummies(new_data.drop(columns=['Churn']))
y = new_data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy on training and test sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Classification report
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1
)

# Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='g', alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves (RandomForestClassifier)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'\nCross-validation Accuracy Scores: {cv_scores}')
print(f'Average Cross-validation Accuracy: {cv_scores.mean():.4f}')


# In[91]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')


# In[108]:


model_grid = RandomForestClassifier(max_depth= 15, max_features= 'sqrt', max_leaf_nodes= None, min_samples_leaf = 1, min_samples_split= 2)
    
model_grid.fit(X_train, y_train) 
y_pred_grid = model_grid.predict(X_test) 
print(classification_report(y_pred_grid, y_test)) 


# In[ ]:




