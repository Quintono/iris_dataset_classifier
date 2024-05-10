# Data analysis libararies
import pandas as pd
import numpy as np

# Data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

iris = pd.read_csv("iris_unlabeled.csv")

# PART 1: check data for information

# Number of observations and missing values
# print(iris.info())
# Check basic description for features
# print(iris.drop(['species'], axis=1).describe())
# Check the response variable frequency
# print(iris['species'].value_counts())

# PART 2: Exploratory data analysis

# Create a pairplot of the data set. Which flower species seems to be the most separable?
#sns.pairplot(iris, hue='species')
#plt.show()
# Setosa seems most seprable from the other two species

# PART 3: Train Test Split

# Split data into a traning set and a testing set.
# train_test_split shuffle the data before the split (shuffle=True by default)
X=iris.drop(['species'], axis=1)
y=iris['species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, shuffle=True, random_state=100)

# PART 4: Train a Model

# Now it's time to train a Support Vector Machine Classifier
# Call SVC() model from sklearn and fit the model to the training data
model = SVC(C=1, kernel='rbf', tol=0.001)
model.fit(X_train, y_train)

# PART 5: Model Evalutation
# Now get predictions from the model and create a confusion matrix and classification report
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score is: ', accuracy_score(y_test, pred))