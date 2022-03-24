import sklearn
from sklearn.datasets import load_iris

iris = load_iris()
##print(iris)
#print('The data matrix:\n',iris['data'])
#print('The classification target:\n',iris['target'])
#print('The names of the dataset columns:\n',iris['feature_names'])
#print('The names of target classes:\n',iris['target_names'])
#print('The full description of the dataset:\n',iris['DESCR'])
#print('The path to the location of the data:\n',iris['filename'])



trainingSetSetosa = iris['data'][0:29]
print(len(trainingSetSetosa))
print()
testingSetSetosa = iris['data'][30:50]
print(len(testingSetSetosa))

trainingSetVersicolor = iris['data'][51:79]
testingSetVersicolor = iris['data'][80:99]

trainingSetVirginica = iris['data'][100:129]
testingSetVirginica = iris['data'][130:149]

