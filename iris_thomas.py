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
    plt.title('Petal data')
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend()
    plt.show()
    return

#plot_petal_data(iris['data'])

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


#-----------------------------------------------



trainingSetSetosa = iris['data'][0:30]
testingSetSetosa = iris['data'][30:50]
trainingSetVersicolor = iris['data'][50:80]
testingSetVersicolor = iris['data'][80:100]
trainingSetVirginica = iris['data'][100:130]
testingSetVirginica = iris['data'][130:150]
#The total training set is now a 90x4 (Rows x Columns) matrix
totalTrainingSet = np.concatenate((trainingSetSetosa,trainingSetVersicolor,trainingSetVirginica), axis=0)
totalTrainingSet = np.reshape(totalTrainingSet,[30*Num_Classes,Num_Features])

#print("total training set: ", totalTrainingSet)

totalTestingSet = np.concatenate((testingSetSetosa,trainingSetVersicolor,testingSetVirginica), axis=0)
print("total testing set: ", totalTestingSet)
#totalTestingSet = np.reshape(totalTestingSet,[20*Num_Classes,Num_Features])


print("total testing set: ", totalTestingSet)

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

# print("target list[0]", get_Targets()[0])       ---> SETOSA
# print("target list[30]", get_Targets()[30])        ---> VERSICOLOR
# print("target list[89]", get_Targets()[89])     ---> VIRGINICA


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

alpha = 0.01

#Finds the class which is closest to a given label vector, i.e. [0.32, 0.82, 0.24] -> [0,1,0]
def round_predictions(g):
    g_rounded = g
    for i in range(len(g_rounded)):
        max_val = max(g_rounded[i])
        for k in range(len(g_rounded[0])):
            if(g_rounded[i][k] >= max_val):
                g_rounded[i][k] = 1
            else:
                g_rounded[i][k] = 0           
    return g_rounded.astype(int) #Returns the predictions, here every entry x in the [x,x,x] is changed from float to int


#---- funksjon som mapper g_rounded [0,0,1]-> 2 

#vet ikke helt hvordan implementer, men skrev litt tanker...

#Da kan vi senere bruke sklearn sin train_test_split funkjson. 
#train_test_split(actual_data, predicted_data, random_state = none)
def map_prediction_to_class(#liste med g_rounded.astype(int)):
    prediction = g_rounded.astype(int)
    prediction_set = []
    for i in #liste med avrundede verdier
        if  prediction == get_Targets()[0]:
            prediction = 0
        elif prediction == get_Targets()[30]:
            prediction = 1
        else:
            prediction = 2
        return prediction


#----------------------------------------------------------------------

def error_rate(g,t): 
    error_count = 0
    for i in range(len(g)):
        if not np.array_equal(g[i],t[i]):
            error_count += 1
    return error_count/len(g) #share of wrong predictions

#x are the different training sets / samples




#-------- CONFUSION MATRIX ----
#https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python


actual = #liste med arrays av de sanne verdiene
predicted = #Liste med data av de spådde

#g is still the predictions and t the true labels

#Må gjøre g_rounded.astype(int) til en array?

def generate_confusion_matrix(actual, predicted):

    #extract the different classes
    #unique funkjsonen funker trolig ikke så bra med 
    pred_classes = np.unique(actual)
    #pred_classes = np.unique(g_rounded.astype(int)) #Da skal dette bli 3
    
    #Initialize the confusion matrix
    confusion_matrix = np.zeros((len(pred_classes),len(pred_classes)))

    #Looping across the different combination of actual /predicted  classes
    for i in range(len(pred_classes)):
        for j in range(len(pred_classes)):

            confusion_matrix[i,j] = np.sum((actual == pred_classes[i] & (predicted == pred_classes[j])))
    
    error_rate_pred = (1 - np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100
    print(f"Error rate: {error_rate_pred}%")
    
    return confusion_matrix

#Begge disse skal bli til lister bare med [0,1,2]
actual = [[0,1,0], [0,0,1], [0,0,1], [0,0,1], [1,0,0], [0,0,1], [0,0,1], [1,0,0], [0,1,0]]
predicted = [[0,0,1], [0,0,1], [1,0,0], [0,0,1], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,1,0]]

print(generate_confusion_matrix(actual, predicted))





#---------------------

X = iris['data']
y = iris['target']

# print('test data: ', X)
# print('test target: ',y)

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=None)




# # Plotting the confusion matrices:
# plt.clf()
# plt.cla()
# cm = confusion_matrix(y_test, y_pred)
# cm = cm.astype('float')
# normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
# sns.heatmap(cm, annot=True, cmap="tab20b")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title('Confusion matrix')
# fname=''
# plt.show()
# # plt.savefig(fname)


