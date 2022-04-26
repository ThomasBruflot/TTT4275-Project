from cgi import test
from re import S
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

Num_Classes = 3
Num_Features = 4
iris = load_iris()
##print(iris)
#print('The data matrix:\n',iris['data'])
#print('The classification target:\n',iris['target'])
#print('The names of the dataset columns:\n',iris['feature_names'])
#print('The names of target classes:\n',iris['target_names'])
#print('The full description of the dataset:\n',iris['DESCR'])
#print('The path to the location of the data:\n',iris['filename'])



#----- litt PLOTTINGs -----

def plot_petal_data(vec):
    #param vec is the iris dataset. 

    length = []
    width = []
    l = int(len(vec)/3) #50 hver

    for line in vec:
        length.append(float(line[2]))   #[2] er petal_length
        width.append(float(line[3]))    #[3] er petal_length
    
    plt.scatter(length[0:l],width[0:l], color='r', label=iris['target_names'][0])
    plt.scatter(length[l:2*l], width[l:2*l], color='g', label=iris['target_names'][1])
    plt.scatter(length[2*l:3*l],width[2*l:3*l], color='b', label=iris['target_names'][2])
    plt.title('Petal data', fontsize=16)
    plt.xlabel('Petal length [cm]', fontsize=14)
    plt.ylabel('Petal width [cm]', fontsize=14)
    plt.legend()
    plt.show()
    return

plot_petal_data(iris['data'])

def plot_sepal_data(vec):
    #param vec is the iris dataset. 
    length = []
    width = []
    l = int(len(vec)/3) #50 hver

    for line in vec:
        length.append(float(line[0]))   #[0] er sepal_length
        width.append(float(line[1]))    #[1] er sepal_length
    
    plt.scatter(length[0:l],width[0:l], color='r', label=iris['target_names'][0])
    plt.scatter(length[0:l],width[0:l], color='r', label=iris['target_names'][0])
    plt.scatter(length[l:2*l], width[l:2*l], color='g', label=iris['target_names'][1])
    plt.scatter(length[2*l:3*l],width[2*l:3*l], color='b', label=iris['target_names'][2])
    plt.title('Sepal data')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('sepal width [cm]')
    plt.legend()
    plt.show()
    return

#plot_sepal_data(iris['data'])