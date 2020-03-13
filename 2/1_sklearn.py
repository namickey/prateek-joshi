#encoding:utf-8
import numpy as np
from sklearn import preprocessing
from functools import reduce

def a():
    inputdata = np.array([[5.1, -2.9, 3.3],
    [-1.2, 7.8,-6.1],
    [3.9,0.4,2.1],
    [7.3,-9.9,-4.5]])

    # バイナライズ
    data_binarized = preprocessing.Binarizer(threshold=2.1).transform(inputdata)
    print("Binarized data:\n", data_binarized)

    # 平均値を引く
    print('before:')
    print('mean=', inputdata.mean(axis=0))
    print(reduce((lambda z, y:z+y), inputdata.flatten())/len(inputdata.flatten()))
    print('std deviation=', inputdata.std(axis=0))

    datascaled = preprocessing.scale(inputdata)
    print('after')
    print('mean=', datascaled.mean(axis=0))
    print('std deviation=', datascaled.std(axis=0))

    # スケーリン
    dataScalerMinmax = preprocessing.MinMaxScaler(feature_range=(0,1))
    dataScaledMinmax = dataScalerMinmax.fit_transform(inputdata)
    print('min max scaled\n', dataScaledMinmax)

    # 正規化
    l1 = preprocessing.normalize(inputdata, norm='l1')
    l2 = preprocessing.normalize(inputdata, norm='l2')
    print('l1: normalized data:\n', l1)
    print('l2: normalized data:\n', l2)

def b():
    # ラベルのエンコーディング
    inputLabels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(inputLabels)
    print('label mapping.')
    for i,item in enumerate(encoder.classes_):
        print(item, '-->', i)

    testLabels = ['green','red','black']
    encodedValues = encoder.transform(testLabels)
    print('labes:', testLabels)
    print('values:', encodedValues)

    testValues = [3,0,4,1]
    decodedList = encoder.inverse_transform(testValues)
    print('values:', testValues)
    print('labes:', list(decodedList))

    # ロジスティック回帰による分類気
    
a()
b()
