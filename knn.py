# -*- coding: utf-8 -*-
import numpy as np
import urllib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt

def main():

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    raw_data = urllib.urlopen(url)
    dataset = np.loadtxt(raw_data, dtype="string", delimiter=",")
    le = preprocessing.LabelEncoder()
    le.fit(["vhigh","high","med","low","5more","more","small","big","2","3","4","unacc","acc","good","vgood"])

    dataset = le.transform(dataset)
    X = dataset[:,0:6]
    Y = dataset[:,6]
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree').fit(X_train, Y_train)
    
    NeighborsCount = []
    trainScores = []
    testScores = []

for weights in ['uniform', 'distance']:
    for i in range(1,500):
        clf = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree',weights=weights).fit(X_train,Y_train)
        NeighborsCount.append(i)
        trainScores.append(clf.score(X_train,Y_train))
        testScores.append(clf.score(X_test,Y_test))

    #Plotear lo q corresponda
    plt.figure(1)
    plt.plot(NeighborsCount, trainScores)
    plt.plot(NeighborsCount, testScores)
    plt.title(r"0% ruido")

    #noise_idx = np.random.random(Y_train.shape)
    #Y_train_with_noise = Y_train.copy()
    #Y_train_with_noise[noise_idx<0.3] = np.floor(Y_train_with_noise[noise_idx<0.3] - 1) * (-1)
    plt.show()
    return
    NeighborsCount = []
    trainScores = []
    testScores = []
    for cant_noise_var in [2,5,10,20]
        #Agregar Variables random
        for i in range(1,500):
            clf = KNeighborsClassifier  (n_neighbors=i, algorithm='ball_tree').fit(X_train,Y_train_with_noise)
            NeighborsCount.append(i)
            trainScores.append(clf.score(X_train,Y_train))
            testScores.append(clf.score(X_test,Y_test))

        #Plotear lo q corresponda
        plt.figure(2)
        plt.plot(NeighborsCount, trainScores)
        plt.plot(NeighborsCount, testScores)
        plt.title(r"10 variables random")
        plt.show()

if __name__ == '__main__':
    main()
