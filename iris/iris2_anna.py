import numpy as np
import pandas as pd                     
import matplotlib.pyplot as plt         #visual   
import seaborn as sns
import csv

#balanced dataset

iris = pd.read_csv("/Users/annaandersen/Desktop/EDC_project/TTT4275-Project/iris.csv")

#Printing data-points and teatures
#iris.shape give the shape of the matrix 
print("There are ", iris.shape[0], " datapoints/rows.")
print("There are ", iris.shape[1], " features/columns.")
print(iris.columns)

iris.value_counts()
#print(iris.keys)
print(iris['spicies'].value_counts()) 

# iris.plot(kind='scatter', x="sepal_length", y="sepal_width")
# plt.show()


#--------- funkjson som plotter sepal length vs sepal width ------
# param vec er datasettet

def plot_sepal_data(vec):
    length = []
    width = []
    l = int(len(vec)/3) #50 hver

    for line in vec:
        length.append(float(line[0]))   #[0] er sepal_length
        width.append(float(line[1]))    #[1] er sepal_length
    
    plt.plot(length[0:l],width[0:l], 'r' 'x')
    plt.plot(length[2*l:3*l],width[2*l:3*l], 'r' 'x')
    return

#--------- funkjson som plotter petal length vs sepal width ------
# param vec er datasettet


def plot_petal_data(vec):
    length = []
    width = []
    l = int(len(vec)/3) #50 hver

    for line in vec:
        length.append(float(line[2]))   #[2] er petal_length
        width.append(float(line[3]))    #[3] er petal_length
    
    plt.plot(length[0:l],width[0:l], 'r' 'x')
    plt.plot(length[2*l:3*l],width[2*l:3*l], 'r' 'x')
    plt.show()
    return


