# -*- coding: utf-8 -*-
import numpy as np
import urllib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random as rn

def main():

    NeighborsCount = range(1,600,20)
    random_start = 0
    random_end = 100000
    
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    raw_data = urllib.urlopen(url)
    dataset = np.loadtxt(raw_data, dtype="string", delimiter=",")
    #le = preprocessing.LabelEncoder()
    #le.fit(["vhigh","high","med","low","5more","more","small","big","2","3","4","unacc","acc","good","vgood"])

    #dataset = le.transform(dataset)

    X = dataset[:,0:8]
    #X = preprocessing.normalize(Z,copy=True)
    Y = dataset[:,8]
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2 )#,random_state=0)
    

    trainScores = {}
    testScores = {}
    plt.figure(1)
    plt.title(r"0% ruido")
    plt.ylabel("Score")
    plt.xlabel("Numero de vecinos")
    for weights in ['distance']:#,'uniform']:
        trainScores[weights] = []
        testScores[weights] = []
        for i in NeighborsCount:
            clf = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree',weights=weights).fit(X_train,Y_train)
            trainScores[weights].append(clf.score(X_train,Y_train))
            testScores[weights].append(clf.score(X_test,Y_test))

    #Plotear lo q corresponda
        plt.plot(NeighborsCount, trainScores[weights],label='Train Score '+weights)
        plt.plot(NeighborsCount, testScores[weights],label='Test Score '+weights)

    #noise_idx = np.random.random(Y_train.shape)
    #Y_train_with_noise = Y_train.copy()
    #Y_train_with_noise[noise_idx<0.3] = 1 - Y_train_with_noise[noise_idx<0.3]
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.title("0 ruido")
    plt.ylabel("Score")
    plt.xlabel("Numero de vecinos")

    
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    add_vars = 0
    for cant_noise_var in [0,20]:
        add_vars += cant_noise_var
        #Agregar Variables random
        for i in xrange(0,cant_noise_var):
            for j in xrange(0,len(X_train)):
                X_train[j].append(rn.randint(random_start,random_end))
            for j in xrange(0,len(X_test)):
                X_test[j].append(rn.randint(random_start,random_end)) 
        trainScores = {}
        testScores = {}
        for weights in ['uniform']:#,'uniform']:
            trainScores[weights] = []
            testScores[weights] = []
            for i in NeighborsCount:

                clf = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree').fit(X_train,Y_train)
                trainScores[weights].append(clf.score(X_train,Y_train))
                testScores[weights].append(clf.score(X_test,Y_test))
            plt.plot(NeighborsCount, trainScores[weights],label='Train Score '+weights+' '+str(add_vars))
            plt.plot(NeighborsCount, testScores[weights],label='Test Score '+weights+' '+str(add_vars))

        #Plotear lo q corresponda
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    main()
