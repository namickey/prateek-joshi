import numpy as np
from sklearn import preprocessing

inputdata = np.array([[5.1, -2.9, 3.3],
[-1.2, 7.8,-6.1],
[3.9,0.4,2.1],
[7.3,-9.9,-4.5]])

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(inputdata)
print("Binarized data:\n", data_binarized)
