import sklearn
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

Num_Classes = 3
Num_Features = 4
Num_Data_P_Class = 30
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

def loadData(case):
    if(case == 0):
        trainingSetSetosa = iris['data'][0:30]
        testingSetSetosa = iris['data'][30:50]
        trainingSetVersicolor = iris['data'][50:80]
        testingSetVersicolor = iris['data'][80:100]
        trainingSetVirginica = iris['data'][100:130]
        testingSetVirginica = iris['data'][130:150]
        #The total training set is now a 90x4 (Rows x Columns) matrix
        totalTrainingSet = np.concatenate((trainingSetSetosa,trainingSetVersicolor,trainingSetVirginica), axis=0)
        totalTrainingSet = np.reshape(totalTrainingSet,[30*Num_Classes,Num_Features])
        #The total testing set is now a 60x4 (Rows x Columns) matrix
        totalTestingSet = np.concatenate((testingSetSetosa,testingSetVersicolor,testingSetVirginica), axis=0)
        totalTestingSet = np.reshape(totalTestingSet,[20*Num_Classes,Num_Features])
    else:
        testingSetSetosa = iris['data'][0:20]
        trainingSetSetosa = iris['data'][20:50]
        testingSetVersicolor = iris['data'][50:70]
        trainingSetVersicolor = iris['data'][70:100]
        testingSetVirginica = iris['data'][100:120]
        trainingSetVirginica = iris['data'][120:150]
        #The total training set is now a 90x4 (Rows x Columns) matrix
        totalTrainingSet = np.concatenate((trainingSetSetosa,trainingSetVersicolor,trainingSetVirginica), axis=0)
        totalTrainingSet = np.reshape(totalTrainingSet,[30*Num_Classes,Num_Features])
        #The total testing set is now a 60x4 (Rows x Columns) matrix
        totalTestingSet = np.concatenate((testingSetSetosa,testingSetVersicolor,testingSetVirginica), axis=0)
        totalTestingSet = np.reshape(totalTestingSet,[20*Num_Classes,Num_Features])
    return totalTrainingSet,totalTestingSet

#We need a total array of t values, we thus need to make a function that makes an array containing 
#the targets in iris set.
def get_Targets(n_data_per_class):
    #We here say that the first 30 values in the training set are all [1,0,0] which means
    #[100% setosa, 0%versicolor, 0%virginica] and so on. Therefore t is a 3x90 array
    t = []
    for i in range(n_data_per_class):
        t.append([1,0,0])
    for i in range(n_data_per_class):
        t.append([0,1,0])
    for i in range(n_data_per_class):
        t.append([0,0,1])
    return np.array(t)




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

def calculate_prediction_g(x,W,n_data_per_class):
    g = np.zeros([n_data_per_class*Num_Classes,Num_Classes])
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


def error_rate(g,t): 
    error_count = 0
    for i in range(len(g)):
        if not np.array_equal(g[i],t[i]):
            error_count += 1
    return error_count/len(g) #share of wrong predictions

#x are the different training sets / samples

#-------- CONFUSION MATRIX ----
#g is still the predictions and t the true labels
def calculate_confusion_matrix(g, t): 
    classes = np.unique(g)

    confusion_matrix = np.zeros((3,3)).astype(int)
    for gk,tk in zip(g,t): #Går gjennom både predictions og true labels parallelt
            pred_index = np.argmax(gk) #Gir ut 0 for [1 0 0], 1 for [0 1 0] og 2 for [0 0 1]
            true_index = np.argmax(tk) #Gir ut 0 for [1 0 0], 1 for [0 1 0] og 2 for [0 0 1]
            confusion_matrix[true_index][pred_index] += 1 #Bruker hvor 1-eren er til å indeksere matrisen
                

        
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes, name="Confusin martix"):
    dataframe_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    fidure = plt.figure(num=name, figsize=(5,5))

    sns.heatmap(dataframe_cm, annot=True)
    plt.show()

def training_lin_classifier(trainingSetSamples,trainingSetTrueLabels,alpha, iterations=500):
    W = np.zeros((Num_Classes,Num_Features+1)) #number of classes and number of features as it is CxD and D is dimension for features, have 5 because of w_0
    MSE_List = []
    #Here we start the actual training by iterating through the training set and using the MSE
    for i in range(iterations):
        #training:
        g = calculate_prediction_g(trainingSetSamples,W,Num_Data_P_Class) #Use the totalTrainingSet defined earlier with all the data
        W = calculate_W(W,alpha,calculate_gradW_MSE(g,trainingSetTrueLabels,trainingSetSamples))
        MSE = mean_squared_error(g,trainingSetTrueLabels)
        MSE_List.append(MSE)
    
    er = error_rate(round_predictions(g),trainingSetTrueLabels)
    print(er)
    conf_matr = calculate_confusion_matrix(round_predictions(g),trainingSetTrueLabels)
    print(conf_matr)
    return np.array(MSE_List),g, W

totalTrainingSet, totalTestingSet = loadData(1)

MSE_List_ret, g_ret, W_ret = training_lin_classifier(totalTrainingSet,get_Targets(Num_Data_P_Class),alpha,2000)
print("MSE_LIST and g from testing: ",MSE_List_ret,g_ret)

def testing_lin_classifier(testSetSamples,testSetTrueLabels,W, iterations=500):
    test_MSE_List = []

    for i in range(iterations):
        g = calculate_prediction_g(testSetSamples, W,20)
        MSE = mean_squared_error(g,testSetTrueLabels)
        test_MSE_List.append(MSE)
    er = error_rate(round_predictions(g),testSetTrueLabels)
    print(er)
    conf_matr = calculate_confusion_matrix(round_predictions(g),testSetTrueLabels)
    print(conf_matr)
    return np.array(test_MSE_List),g

test_MSE_List_ret, g_test_ret = testing_lin_classifier(totalTestingSet, get_Targets(20), W_ret,2000)
print("Testing results: \n")
print("MSE_List_test and g_test: ",MSE_List_ret,g_ret)




#------------------ HISTOGRAM -----------------------

def plot_histograms():
    #We have 4 features per class (sep len and width, pet len and width)
    #We need to plot one feature at the time and look at it for the three classes. 
    feature_list = [] #[Seplen,SepWid,PetLen,PetWid]
    num_bins = 10
    #Extract features into the feature list to make it easier to make histogram
    feature_list.append([i[0] for i in iris['data']])
    feature_list.append([i[1] for i in iris['data']])
    feature_list.append([i[2] for i in iris['data']])
    feature_list.append([i[3] for i in iris['data']])

    
    hist0_set,bins0_set = np.histogram(feature_list[0][0:50], bins=10)
    hist0_ver,bins0_ver = np.histogram(feature_list[0][50:100], bins=10)
    hist0_vir,bins0_vir = np.histogram(feature_list[0][100:150], bins=10)

    hist1_set,bins1_set = np.histogram(feature_list[1][0:50], bins=10)
    hist1_ver,bins1_ver = np.histogram(feature_list[1][50:100], bins=10)
    hist1_vir,bins1_vir = np.histogram(feature_list[1][100:150], bins=10)

    hist2_set,bins2_set = np.histogram(feature_list[2][0:50], bins=10)
    hist2_ver,bins2_ver = np.histogram(feature_list[2][50:100], bins=10)
    hist2_vir,bins2_vir = np.histogram(feature_list[2][100:150], bins=10)

    hist3_set,bins3_set = np.histogram(feature_list[3][0:50], bins=10)
    hist3_ver,bins3_ver = np.histogram(feature_list[3][50:100], bins=10)
    hist3_vir,bins3_vir = np.histogram(feature_list[3][100:150], bins=10)

    plt.subplot(2,3,1)
    plt.title("Histogram of 4 features for Setosa")
    plt.xlabel("Measurement [cm]")
    plt.ylabel("Number of samples in bin")
    plt.hist(feature_list[0][0:50], bins=10)
    plt.hist(feature_list[1][0:50], bins=10)
    plt.hist(feature_list[2][0:50], bins=10)
    plt.hist(feature_list[3][0:50], bins=10)

    plt.subplot(2,3,2)
    plt.title("Histogram of 4 features for Versicolor")
    plt.xlabel("Measurement [cm]")
    plt.ylabel("Number of samples in bin")
    plt.hist(feature_list[0][50:100], bins=10)
    plt.hist(feature_list[1][50:100], bins=10)
    plt.hist(feature_list[2][50:100], bins=10)
    plt.hist(feature_list[3][50:100], bins=10)

    plt.subplot(2,3,3)
    plt.title("Histogram of 4 features for Virginica")
    plt.xlabel("Measurement [cm]")
    plt.ylabel("Number of samples in bin")
    plt.hist(feature_list[0][100:150], bins=10)
    plt.hist(feature_list[1][100:150], bins=10)
    plt.hist(feature_list[2][100:150], bins=10)
    plt.hist(feature_list[3][100:150], bins=10)


    #create legend
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [Sepal-Length,Sepal-Width, Petal-Length,Petal-Width]]
    #labels= ["Sepal-Length","Sepal-Width", "Petal-Length","Petal-Width"]
    #plt.legend(handles, labels)
    

    plt.show()
    print(hist0_set)
    print(bins0_set)

plot_histograms()





