import numpy as np
from sklearn.datasets import load_iris


#data = load_iris()              #load_iris() is a link that provides documentation
iris = load_iris()

print('The data matrix:\n',iris['data'])
print('The classification target:\n',iris['target'])
print('The names of the dataset columns:\n',iris['feature_names'])
print('The names of target classes:\n',iris['target_names'])


annatull = 1