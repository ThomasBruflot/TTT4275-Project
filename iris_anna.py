import pandas as pd                     
import numpy as np
import matplotlib.pyplot as plt         #visual   
import seaborn as sns                   #Simple graphmodel
from sklearn.datasets import load_iris


#data = load_iris()              #load_iris() is a link that provides documentation

#https://en.wikipedia.org/wiki/Iris_flower_data_set
#The data set contains 3 classes of 50 instances each, where each class refert to a type of iris plant.
#One class is linearly separable from the other 2. The latter /ramainin g2) are NOT linearly seperable from each other

iris = load_iris()

# print('The data matrix:\n',iris['data'])
# print('The classification target:\n',iris['target'])
# print('The names of the dataset columns:\n',iris['feature_names'])
# print('The names of target classes:\n',iris['target_names'])


#print(iris.keys())

#print(iris["DESCR"]) #Description of the dataset represented in a nice way

#print(iris.data.T) #plot the transpose. List of lists. Print 4 sublists are arrays that corr esponds to 1.sepal length, 2.sepal width, 3. petal length, 4. petal width 

features = iris.data.T

sepal_length = features[0]
spetal_width = features[1]
petal_length = features[2]
petal_width = features[3]


sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

print(sepal_length_label)


 