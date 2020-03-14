#encoding:utf-8
import numpy as np
from sklearn import preprocessing
from functools import reduce
import matplotlib.pyplot as plt

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
    from sklearn import linear_model
    X = np.array([[3.1, 7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],
    [5.6,5],[3.3,0.4],[3.9,0.9],[2.8,1],[0.5,3.4],
    [1,4],[0.6,4.9]])
    y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
    print(len(X), len(y))
    classifier = linear_model.LogisticRegression(solver='liblinear', C=1, multi_class='auto')
    classifier.fit(X, y)
    vc(classifier, X, y)

# visualize_classifier
def vc(classifier, X, y, title=''):
    print(X[:, 0])
    minx, maxx = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    miny, maxy = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    print(minx, maxx)
    print(miny, maxy)
    mesh_step_size = 0.1
    xvals, yvals = np.meshgrid(np.arange(minx, maxx, mesh_step_size),
                               np.arange(miny, maxy, mesh_step_size))
    output = classifier.predict(np.c_[xvals.ravel(), yvals.ravel()])
    print(output)
    print(xvals.shape)
    output = output.reshape(xvals.shape)
    print(output)
    plt.figure()
    plt.title(title)
    plt.pcolormesh(xvals, yvals, output, cmap=plt.cm.Set3)
    plt.scatter(X[:,0], X[:,1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.jet)
    plt.xlim(xvals.min(), xvals.max())
    plt.ylim(yvals.min(), yvals.max())
    plt.xticks((np.arange(int(minx), int(maxx), 1.0)))
    plt.yticks((np.arange(int(miny), int(maxy), 1.0)))
    plt.show()

def c():
    #単純ベイズ分類器
    from sklearn.naive_bayes import GaussianNB

#a()
#b()
c()
