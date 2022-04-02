import sklearn
from sklearn.datasets import load_iris
import numpy as np


Num_Classes = 3
Num_Features = 4
iris = load_iris()
##print(iris)
#print('The data matrix:\n',iris['data'])
print('The classification target:\n',iris['target'])
#print('The names of the dataset columns:\n',iris['feature_names'])
#print('The names of target classes:\n',iris['target_names'])
#print('The full description of the dataset:\n',iris['DESCR'])
#print('The path to the location of the data:\n',iris['filename'])
trainingSetSetosa = iris['data'][0:30]
testingSetSetosa = iris['data'][30:50]
trainingSetVersicolor = iris['data'][50:80]
testingSetVersicolor = iris['data'][80:100]
trainingSetVirginica = iris['data'][100:130]
testingSetVirginica = iris['data'][130:150]
#The total training set is now a 90x4 (Rows x Columns) matrix
totalTrainingSet = np.concatenate((trainingSetSetosa,trainingSetVersicolor,trainingSetVirginica), axis=0)
totalTrainingSet = np.reshape(totalTrainingSet,[30*Num_Classes,Num_Features])
#We need a total array of t values, we thus need to make a function that makes an array containing 
#the targets in iris set.
def get_Targets():
    N = 30
    #We here say that the first 30 values in the training set are all [1,0,0] which means
    #[100% setosa, 0%versicolor, 0%virginica] and so on. Therefore t is a 3x90 array
    t = []
    for i in range(N):
        t.append([1,0,0])
    for i in range(N):
        t.append([0,1,0])
    for i in range(N):
        t.append([0,0,1])
    return np.array(t)
print("target list: ", get_Targets())



#Chapter 3.2 describes the MSE as a way to distinguish between the different iris types as it
#is a linear seperable problem

# g is a vector containing predicted labels
# t is a vector containing true labels
#Compendium eq. (19)
def calculate_MSE(g,t):
    error = g-t #dimensions 3x90 and 3x90
    error_transposed = np.transpose(g-t)
    mseVal = 1/2*np.sum(np.matmul(error,error_transposed)) 
    return mseVal

# Here x is a single input 
# W is a CxD matrix where C is number of different classes 
#Comp. eq. (20)
def sigmoid(x):
    return np.array(1/(1+np.exp(-x)))

def calculate_prediction_g(x,W):
    g = np.zeros([30*Num_Classes,Num_Classes])
    for i,sample in enumerate(x):
        sample = np.append([sample],[1])
        z = np.matmul(W,sample)
        g[i] = sigmoid(z)
    return g

#Need to implement equation (22) and (23)
#Comp. Eq. (22):
def calculate_gradW_MSE(g,t,x):
    gradW_MSE = np.zeros([Num_Classes,Num_Features+1])
    for xk,gk,tk in zip(x,g,t):    
        xk = np.append([xk],[1]) #Lager basically en ny matrise med enere i siste del, 3x4->3x5 med 1 på slutten
        xk = xk.reshape(Num_Features+1,1) #Her blir så dette transponert
        #Under her implementeres grad_gk_MSE og så grad_zk_g 
        grad_zk_g = (np.ones((Num_Classes,1))-gk.reshape(Num_Classes,1)) * (gk.reshape(Num_Classes,1))
        grad_gk_MSE = ((gk-tk)).reshape(Num_Classes,1)
        tmp = grad_gk_MSE * grad_zk_g
        #Tar her og regner ut 
        gradW_MSE += np.matmul(tmp, xk.reshape(1,Num_Features+1))
    return gradW_MSE

#Comp. Eq. (23):
def calculate_W(prevW,alpha,gradW_MSE):
    W = prevW-alpha*gradW_MSE
    return W

alpha = 0.5

#x are the different training sets / samples

def training_lin_classifier(trainingSetSamples,trainingSetTrueLabels,alpha, iterations=500):
    W = np.zeros((Num_Classes,Num_Features+1)) #number of classes and number of features as it is CxD and D is dimension for features, have 5 because of w_0
    MSE_List = []
    #Here we start the actual training by iterating through the training set and using the MSE
    for i in range(iterations):
        #training:
        g = calculate_prediction_g(trainingSetSamples,W) #Use the totalTrainingSet defined earlier with all the data
        W = calculate_W(W,alpha,calculate_gradW_MSE(g,trainingSetTrueLabels,trainingSetSamples))
        MSE = calculate_MSE(g,trainingSetTrueLabels)
        MSE_List.append(MSE)
    return np.array(MSE_List)

print("MSE_LIST: ", training_lin_classifier(totalTrainingSet,get_Targets(),alpha,100))

