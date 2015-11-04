# -*- coding: utf-8 -*-
import sys
import numpy as np
import urllib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random as rn

def main(weights):

    NeighborsCount = range(1,200,5)
    random_start = 0
    random_end = 100
    
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    raw_data = urllib.urlopen(url)
    dataset = np.loadtxt(raw_data, dtype="float", delimiter=",")
    
    X = dataset[:,0:8]
    X = X / X.sum(axis=0)
    Y = dataset[:,8]
    
    random_seeds = range(0,100,5)
    for seed in random_seeds:
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2 ,random_state=seed)
        trainScores = {}
        testScores = {}
        plt.figure()
        plt.title(r"KNN con " + weights)
        plt.ylabel("Score")
        plt.xlabel("Numero de vecinos")
##      for weights in ['distance','uniform']:
        trainScores[weights] = []
        testScores[weights] = []
        for i in NeighborsCount:
            clf = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree',weights=weights).fit(X_train,Y_train)
            trainScores[weights].append(clf.score(X_train,Y_train))
            testScores[weights].append(clf.score(X_test,Y_test))

    #Plotear lo q corresponda
        plt.plot(NeighborsCount, trainScores[weights],label='Train Score')
        plt.plot(NeighborsCount, testScores[weights],label='Test Score')
        plt.legend()

        file_path = "../informe/knn/OriginalSample/KnnwithSeed" + str(seed) +"andWeight" + weights
        plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)

        plt.figure()
        plt.title("KNN con variables aleatorias")
        plt.ylabel("Score")
        plt.xlabel("Numero de vecinos")

        
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        add_vars = 0
        for cant_noise_var in [0,10,10]:
            add_vars += cant_noise_var
            #Agregar Variables random
            for i in xrange(0,cant_noise_var):
                for j in xrange(0,len(X_train)):
                    X_train[j].append(rn.randint(random_start,random_end))
                for j in xrange(0,len(X_test)):
                    X_test[j].append(rn.randint(random_start,random_end)) 
            trainScores = {}
            testScores = {}
##          for weights in ['distance','uniform']:
            trainScores[weights] = []
            testScores[weights] = []
            for i in NeighborsCount:
                clf = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree').fit(X_train,Y_train)
                trainScores[weights].append(clf.score(X_train,Y_train))
                testScores[weights].append(clf.score(X_test,Y_test))

            #plt.plot(NeighborsCount, trainScores[weights],label='Train Score '+weights+' '+str(add_vars))
            plt.plot(NeighborsCount, testScores[weights],label=str(add_vars)+' variables random')

            #Plotear lo q corresponda
        plt.legend()
        
        file_path = "../informe/knn/WithRandomVariables/KnnwithSeed" + str(seed) +"andWeight" + weights
        plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print 'Usage: python knn.py [weights]'
    else:
        main(sys.argv[1])
