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
''


trainingSetSetosa = iris['data'][0:30]
print(len(trainingSetSetosa))
testingSetSetosa = iris['data'][30:50]
print(len(testingSetSetosa))

trainingSetVersicolor = iris['data'][50:80]
print(len(trainingSetVersicolor))
testingSetVersicolor = iris['data'][80:100]
print(len(testingSetVersicolor))

trainingSetVirginica = iris['data'][100:130]
print(len(trainingSetVirginica))
testingSetVirginica = iris['data'][130:150]

print(len(testingSetVirginica))