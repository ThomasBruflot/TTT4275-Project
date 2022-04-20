from ast import Num
from this import d
import numpy as np
import math
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans

#------------

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print((train_x, train_y))
#print('x_train: ' + str(train_x.shape))
#print('x_train_0: ' + str(train_x[0]))
#print('y_train: ' + str(train_y.shape))
print('y_train_5: ' + str(train_labels[5]))
#print('x_test:  '  + str(test_x.shape))
#print('y_test:  '  + str(test_y.shape))
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
    #print(Euclidean_distance)
    return Euclidean_distance

print("test Euclidean dist: ")
calculate_euclidean_distance2(a,b) 


# ----- Implementasjon av nearest neighbour -----

def calculate_nearest_neighbour(trainImages, trainLabels, testImages, testLabels, NumTrain, NumTest):
    #Idea is: We have testImages which we iterate through and find the difference to all
    #the trainImages. We save the difference for each iteration and compare to previous difference
    #The smallest difference is the one we want. We save the index for the one with 
    #smallest difference and save the prediction of that spesific index. Then we can check if correct or not
    wrongPred = []
    corrPred = [] 
    for i in range(NumTest):
        pred = 0 # New prediction for each testImage in the dataset
        indexPred = 0 #Index of prediction
        minDist = 0
        for k in range(NumTrain):
            dist = calculate_euclidean_distance2(testImages[i],trainImages[k])
            if(k == 0):
                minDist = dist # We dont have a minimum difference first iteration
            if(dist < minDist):
                minDist = dist
                indexPred = k #For each time we find a smaller difference we say ok this is our prediction
                pred = trainLabels[k]
        if(pred == testLabels[i]):
            corrPred.append([i,indexPred])
            #Here we can increment confusion matrix at right indexes
        else:
            wrongPred.append([i,indexPred])
    return corrPred,wrongPred

corrPred, wrongPred = calculate_nearest_neighbour(train_images,train_labels,test_images,test_labels,60000,10000)

testVal = 1
print("CorrPred0: ", corrPred[testVal])
print("prediction values: ", test_labels[corrPred[testVal][0]], " ", train_labels[corrPred[testVal][1]])
print("prediction values: ", test_labels[1], " ", train_labels[360])
print("All wrong predictions: ", str(len(wrongPred)))
print("All correct predictions: ", str(len(corrPred)))








