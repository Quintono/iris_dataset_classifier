import pandas as pd

iris = pd.read_csv('iris.csv')

print(iris.head(10))

print(iris.groupby('species').size())
print(iris.describe())