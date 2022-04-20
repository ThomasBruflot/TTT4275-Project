#https://pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/


#--- God motivasjonsvideo
#https://www.youtube.com/watch?v=aircAruvnKk


import numpy as np
import math
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans

from collections import Counter
from sklearn.datasets import fetch_openml
import pandas as pd
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#------------

(train_x, train_y), (test_x, test_y) = mnist.load_data()

print((train_x, train_y))
print('x_train: ' + str(train_x.shape))
print('y_train: ' + str(train_y.shape))
print('x_test:  '  + str(test_x.shape))
print('y_test:  '  + str(test_y.shape))
#------------


a =np.array((1,2,3))
b=np.array((4,5,6))

def calculate_euclidean_distance1(input_1,input_2, length):
    distance = 0
    for i in range(length-1):
        distance += (input_1[i]-input_2[i])**2
        Euclidean_distance = math.sqrt(distance)
    return Euclidean_distance

def calculate_euclidean_distance2(input_1,input_2):
    Euclidean_distance = np.linalg.norm(input_1 - input_2)
    print(Euclidean_distance)
    return Euclidean_distance

print("test Euclidean dist: ")
calculate_euclidean_distance2(a,b) 