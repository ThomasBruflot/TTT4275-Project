from ast import Num
import enum
from this import d
import numpy as np
import math
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from scipy.spatial import distance
import seaborn as sns

#------------

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

flat_train_images = train_images.flatten()
reshap_train_images = np.reshape(train_images, (len(train_images),784))

flat_test_images = test_images.flatten()
reshap_test_images = np.reshape(test_images, (len(test_images),784))

#print("train_images_0: ", train_images[0])
#print("flattened train_images_0: ", flat_train_images[0:784])
#print("reshaped train_images_0: ", reshap_train_images[0:2])
#print((train_x, train_y))
#print('x_train: ' + str(train_x.shape))
#print('x_train_0: ' + str(train_x[0]))
#print('y_train: ' + str(train_y.shape))
#print('y_train_5: ' + str(train_labels[5]))
#print('x_test:  '  + str(test_x.shape))
#print('y_test:  '  + str(test_y.shape))
#------------

def error_rate(g,t): 
    error_count = 0
    for i in range(len(g)):
        if not np.array_equal(g[i],t[i]):
            error_count += 1
    return error_count/len(g) #share of wrong predictions


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

#print("test Euclidean dist: ")
#calculate_euclidean_distance2(a,b) 


# ----- Implementasjon av nearest neighbour -----

def calculate_nearest_neighbour(trainImages, trainLabels, testImages, testLabels, NumTrain, NumTest):
    #Idea is: We have testImages which we iterate through and find the difference to all
    #the trainImages. We save the difference for each iteration and compare to previous difference
    #The smallest difference is the one we want. We save the index for the one with 
    #smallest difference and save the prediction of that spesific index. Then we can check if correct or not
    confusionMatrix = np.zeros((10,10))
    wrongPred = []
    corrPred = [] 
    for i in range(NumTest):
        pred = 0 # New prediction for each testImage in the dataset
        indexPred = 0 #Index of prediction
        minDist = 0
        for k in range(NumTrain):
            dist = calculate_euclidean_distance2(testImages[i],trainImages[k])
            #dist = distance.euclidean(testImages[i],trainImages[k])
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
        confusionMatrix[testLabels[i],pred] += 1
    print("Confusion matrix: \n", confusionMatrix)   
    #-------plot av confusion matrix ---------
    ax = sns.heatmap(confusionMatrix, annot=True, cmap='Greens', linewidths=.7, cbar_kws={"shrink": .82},linecolor='black', clip_on=False)
    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted digit')
    ax.set_ylabel('Actual digit')

    ## Ticket labels - ( must be alphabetical order)
    ax.xaxis.set_ticklabels(['0','1', '2', '3', '4','5','6','7','8','9'])
    ax.yaxis.set_ticklabels(['0','1', '2', '3', '4','5','6','7','8','9'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    unique, counts = np.unique(test_labels, return_counts=True)
    percentMatrix = confusionMatrix
    for i in range(len(percentMatrix)):
        percentMatrix[i] = percentMatrix[i]/counts[i]
    
    ax = sns.heatmap(percentMatrix, annot=True, 
            fmt='.2%', cmap='Greens', linewidths=.7, cbar_kws={"shrink": .82}, linecolor='black', clip_on=False)

    ax.set_title('Confusion Matrix given in percentage\n\n')
    ax.set_xlabel('\nPredicted digit')
    ax.set_ylabel('Actual digit')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1', '2', '3', '4','5','6','7','8','9'])
    ax.yaxis.set_ticklabels(['0','1', '2', '3', '4','5','6','7','8','9'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    
    return corrPred,wrongPred

#corrPred, wrongPred = calculate_nearest_neighbour(reshap_train_images,train_labels,reshap_test_images,test_labels,60000,100)
#
#testVal = 1
#print("CorrPred0: ", corrPred[testVal])
#print("prediction values: ", test_labels[corrPred[testVal][0]], " ", train_labels[corrPred[testVal][1]])
#print("prediction values: ", test_labels[1], " ", train_labels[360])
#print("All wrong predictions: ", str(len(wrongPred)))
#print("All correct predictions: ", str(len(corrPred)))




# ------------- Clustering -------------

def cluster(trainImages, trainLabels, M):
    numDigits = 10
    sortedList = [] #3d array where you have arrays for each of the digits from 0 - 9 sorted with all images corresponding to the value 
    clusterList = []
    newLabels = []

    for i in range(numDigits):
        tmp = []
        for k,label in enumerate(trainLabels):
            if i == label:
                tmp.append(trainImages[k])
        sortedList.append(tmp)


    for i in range(numDigits):
        digitImages = sortedList[i] # digitImages is set to the arrays of images for 0,1,2,3,4,5,6,7,8,9 in order
        clust = KMeans(n_clusters=M).fit(digitImages)
        clusterList.append(clust.cluster_centers_)
        newLabels.append([i]*M) #add 64 labels for each digit because we now have 64 clusters per digit

    clusterArray = np.array(clusterList)
    cluster_templates = np.reshape(clusterArray.flatten(), (M*numDigits,784)) #We first flatten and change the dimensions to be an image (instead of 28*28 we do 1*784)
    cluster_labels = np.reshape(np.array(newLabels).flatten(), (M*numDigits,1))

    return cluster_templates, cluster_labels #Her returneres clusterimages som 640 bilder med 784 verdier 64 per tall, cluster 640 labels med 1 verdi (tallet som samsvarer med bildet)

cluster_templates, cluster_labels = cluster(reshap_train_images,train_labels,64)
print("clusterImage: \n",len(cluster_templates))

corrPred, wrongPred = calculate_nearest_neighbour(cluster_templates,cluster_labels,reshap_test_images,test_labels,640,10000)

testVal = 1
print("CorrPred0: ", corrPred[testVal])
print("prediction values: ", test_labels[corrPred[testVal][0]], " ", train_labels[corrPred[testVal][1]])
print("prediction values: ", test_labels[1], " ", train_labels[360])
print("All wrong predictions: ", str(len(wrongPred)))
print("All correct predictions: ", str(len(corrPred)))



# ------------- K nearest :D:D:D:D:D:D


#unique, counts = np.unique(test_labels, return_counts=True)

#result = np.column_stack((unique, counts)) 
#print(result)

#Test label number of each
##[[   0  980]
# [   1 1135]
# [   2 1032]
# [   3 1010]
# [   4  982]
# [   5  892]
# [   6  958]
# [   7 1028]
# [   8  974]
# [   9 1009]]

