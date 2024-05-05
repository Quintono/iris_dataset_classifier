import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = pd.read_csv('iris.csv')
#print(iris.groupby('species').size())

setosa = iris[iris.species == "setosa"]
versicolor = iris[iris.species == "versicolor"]
virginica = iris[iris.species == "virginica"]

#fig: holds all plot elements
#ax: array of axes
fig, ax = plt.subplots()
fig.set_size_inches(13, 7) # adjusting the length and width of plot

ax.scatter(setosa['petal_length'], setosa['petal_width'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['petal_length'], versicolor['petal_width'], label="Versicoloer", facecolor="green")
ax.scatter(virginica['petal_length'], virginica['petal_width'], label="Virginica", facecolor="red")

ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()

#plt.show()

# Droping the target and speciies since we only need the measurements
X = iris.drop(['target', 'species'], axis=1)
y = iris['target']

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

training_prediction = log_reg.predict(X_train)
print(training_prediction)

test_prediction = log_reg.predict(X_test)
print(test_prediction)

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recal scores
print(metrics.classification_report(y_train, training_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

print("Precision, Recall, Confusion matrix, in testing\n")

# Precision Recal scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))