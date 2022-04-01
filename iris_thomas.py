import sklearn
from sklearn.datasets import load_iris
import numpy as np

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
totalTrainingSet = [trainingSetSetosa,trainingSetVersicolor,trainingSetVirginica]

#Chapter 3.2 describes the MSE as a way to distinguish between the different iris types as it
#is a linear seperable problem

# g is a vector containing predicted labels
# t is a vector containing true labels
#Compendium eq. (19)
def calculate_MSE(g,t):
    error = g-t
    error_transposed = np.transpose(g-t)
    mseVal = 1/2*np.sum(np.matmul(error,error_transposed)) 
    return mseVal

# Here x is a single input 
# W is a CxD matrix where C is number of different classes 
#Comp. eq. (20)
def calculate_prediction_g(x,W):
    expList = []
    #Loop through the samples and multiply with the W weights and use sigmoid to
    #get predictions
    for i in range(x): 
        expList.append(np.exp(-(np.matmul(x[i],W))))
    z_ikExponential = np.array(expList)
    g = 1 / (1+ z_ikExponential)

#Need to implement equation (22) and (23)
#Comp. Eq. (22):
def calculate_gradW_MSE(g,t,x):
    error = g-t
    tmp = 1-g
    tmp2 = np.multiply(error,tmp)
    tmp3 = np.multiply(tmp2,g)
    gradW_MSE = np.sum(np.matmul(tmp3,np.transpose(x)))
    return gradW_MSE

#Comp. Eq. (23):
def calculate_W(prevW,alpha,gradW_MSE):
    W = prevW-alpha*gradW_MSE
    return W

alpha = 0.5

#x are the different training sets / samples
print("W: ", W)
def training_lin_classifier(trainingSetSamples,trainingSetTrueLabels,alpha,t, iterations=500):
    W = np.zeros((3,4)) #number of classes and number of features as it is CxD and D is dimension for features
    MSE_List = []
    #Here we start the actual training by iterating through the training set and using the MSE
    for i in range(iterations):
        #training:
        g = calculate_prediction_g(trainingSetSamples,W) #Use the totalTrainingSet defined earlier with all the data
        W = calculate_W(W,alpha,calculate_gradW_MSE(g,trainingSetTrueLabels,trainingSetSamples))
        MSE = calculate_MSE(g,trainingSetTrueLabels)
        MSE_List.append(MSE)
    return MSE_List