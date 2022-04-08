import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

#Reading the data from irs.csv file
features = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv("/Users/annaandersen/Desktop/EDC_project/TTT4275-Project/iris.csv", names = features)
dataset.columns = features

Classes = 3

#print(dataset.dtypes)  #finding what datatype is in the dataset
#print(dataset.groupby('class').size())  #Number of instances in each class



def load_irisData():
    iris_data = [
        pd.read_csv('/Users/annaandersen/Desktop/EDC_project/TTT4275-Project/iris.data', names = features)]
    print(iris_data)
    return 

load_irisData()




# Confusion matrix
# disp = metrics.plot_confusion_matrix(mod_dt, C_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)
# disp.ax_.set_title('Decision Tree Confusion matrix, without normalization')