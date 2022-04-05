from tempfile import tempdir
import tempfile
import numpy as np
import pandas as pd                     
import matplotlib.pyplot as plt         #visual   
import seaborn as sns
import csv
import copy
from sklearn.datasets import load_iris

#There are 3 classes of iris

Classes = 3
variants = ['Setosa', 'Versicolour', 'Virginicia']

iris = load_iris()

def load_data():
#     #Loads the iris dataset from file and returns a array witg 6 colums
#     #First four are the features, the 5th is a colum of ones, and the last is of the labels

    for i in range(Classes):
        tempfile = np.loadtxt("/Users/annaandersen/Desktop/EDC_project/TTT4275-Project/class_"+str(i+1),delimiter=",")

        class_number = np.ones((tempfile.shape[0],2))   #adding the class, and the column with 1. .shape[0] is rows, and 2 is columns
        class_number[:,-1] *= i 

        tempfile = np.hstack((tempfile, class_number)) #Stack arrays in sequence horizontally (column wise).
        if i > 0:
            data = np.vstack((data, tempfile))         #Stack arrays in sequence vertically (row wise).
        else:
            data = copy.deepcopy(tempfile)             #Any changes made to a copy of object do not reflect in the original object.

    tempfile = data[:,:-1]      #Gives all except last column, in all rows
    tempfile = tempfile / tempfile.max(axis=0)
    data[:,:-1] = tempfile

    return data



# def load_data():
#     total_vec = []
#     training_vec = []
#     testing_vec = []
    
#     for i in range(Classes):
#         tempfile = "/Users/annaandersen/Desktop/EDC_project/TTT4275-Project/class_"+str(i+1)
#         with open(tempfile, 'r') as class_file:
#             data = csv.reader(class_file, delimiter = ',')
#             i = 0
#             for line in data:
#                 total_vec.append(line)
#                 if i < 30:
#                     training_vec.append(line)
#                 else:
#                     testing_vec.append(line)
#                 i += 1

#     for line in total_vec:
#         new_line = []
#         for feature in line:
#             new_line.append(float(feature))
#         new_line.append(1)
#         total_vec.append(new_line)

#     for line in training_vec:
#         new_line = []
#         for feature in line:
#             new_line.append(float(feature))
#         new_line.append(1)
#         training_vec.append(new_line)

    
#     for line in testing_vec:
#         new_line = []
#         for feature in line:
#             new_line.append(float(feature))
#         new_line.append(1)
#         testing_vec.append(new_line)

#     return 



def split_data(data, training_size):
    #Splitting the iris data into training and testing set. 
    #Arguments:
    #data is a array
    #training_size: int, number of trainin samples for the data

    #Returns both trainin data and test data

    N = int(data.shape[0] / Classes)        #data.shape = 150
    test_size = N - training_size
    sample_length = data.shape[1]       #6
    training_data = np.zeros((Classes*training_size, sample_length))
    test_data = np.zeros((Classes*test_size, sample_length))
    for i in range(Classes):
        tempfile = data[(i*N):((i+1)*N)]
        training_data[(i*training_size):((i+1)*training_size),:] = tempfile[:training_size,:]
        test_data[(i*test_size):((i+1)*test_size),:] = tempfile[training_size:,:]

    return training_data, test_data


data = load_data()
#print(data.shape[1])


def plot_petal(data):
    #data is a vector, containing the petal length and width

    for i in range(Classes):
        petal_data = data[(50*i):(50*(i+1)), 2: -2]         #using the features in 3 and 4, of 6 colums
        plt.scatter(petal_data[:,0], petal_data[:,1])
    plt.title('Petal data')
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.show()

def plot_sepal(data):
    #data is a vector, containing the petal length and width

    for i in range(Classes):
        sepal_data = data[(50*i):(50*(i+1)), : -4]         #using the features in 1 and 2, of 6 colums
        plt.scatter(sepal_data[:,0], sepal_data[:,1])
    plt.title('Sepal data')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('sepal width [cm]')
    plt.show()



#--- defining equations from the compendium


#Eq.20
def sigmoid(z_ik):
    #z_ik: array with datapoints
    sigmoid = np.array(1/1(1+np.exp(-z_ik)))       
    return  sigmoid


#Implement eq. 21-23 in compendium

def training_W(data, iterations, alpha):
    #Trains the lin.classifier on the provided dataset
    #data: array with features and labels
    #iterations: number of itterations for the training
    #alpha: int, step factor/size for training
   
    #returns W-matrix, Cx(D+1)

   mse_list = []
   features = data.shape[1]-2       #4
   g_k = np.zeros(Classes)
   g_k[0] = 1
   t_k = np.zeros((Classes,1))      #shape of the new array. Matrix, 3x1
   W = np.zeros((Classes, features+1))

   for i in range(iterations):
       gradW_MSE = 0
       mse = 0
       for x_k in data:
            temp = np.matmul(W,(x_k[:-1]))[np.newaxis].T         #[np.newaxis] adds a new dimension to an array
            g_k = sigmoid(temp)

            #Extracttarget and update t_k
            t_k *= 0
            t_k[int(x_k[-1]),:] = 1
            tk=t_k[np.newaxis].T 

            #Equation 22.
            grad_gk_mse = np.multiply((g_k - t_k), g_k)     #elementwise multiplication
            grad_zk_g = np.multiply(grad_gk_mse, (1-g_k))
            grad_W_zk += np.dot(grad_zk_g.T, (g_k-t_k))

            #Equation 23
            W = W - alpha*gradW_MSE
            return W


def confusion_matrix(W, test_data):
    #sumarize how each method preformed on the testing data
    #computes the confusion matrix for the classifier and the data 
    # Theory described at page 31 in compendium
    
    # W array, Cx(D+1)
    #test_data array with the data to be classified, with features and labels

    confusion_matrix = np.zeros((Classes,Classes))  #3x3
    for i in range(len(test_data)):
        prediction = int(np.argmax(np.matmul(w,test_data[i,:-1])))
        known_truth = int(test_data[i,-1])
        confusion_matrix[prediction, known_truth] += 1

    print(confusion_matrix)
    error = (1 - np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100
    print(f'Error rate: {error}%')
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes, name="Confusion matrix"):
    #the numbers on the dfiagonal tells us how many times the samples were correctly classified
    dataFrame = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    figure = plt.figure(num=name, figsize=(5,5))                                    #5x5 as in table 1, 2 page 31
    sns.heatmap(dataFrame, annot=True)
    plt.show()






#print("Data: ", load_data())
#print("iris: ", iris)
