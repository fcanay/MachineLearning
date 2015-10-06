import urllib
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation


def main():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    raw_data = urllib.urlopen(url)
    dataset = np.loadtxt(raw_data, dtype="float", delimiter=",")
    
    X = dataset[:,0:8]
    Y = dataset[:,8]
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

    random_seeds = range(0,100,5)
    for seed in random_seeds:
        
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2,random_state=seed)


        trainScores = []
        testScores = []
        max_nodes = range(2,150,1)
        for i in max_nodes:
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train, Y_train)
            trainScores.append(clf.score(X_train,Y_train))
            testScores.append(clf.score(X_test,Y_test))

        plt.figure()
        plt.title("Sobreajuste en IDT")
        plt.ylabel("Score")
        plt.xlabel("Numero de nodos maximos")
        plt.plot(max_nodes, trainScores,label='Train Score')
        plt.plot(max_nodes, testScores,label='Test Score') 

        plt.legend()
        file_path = "../informe/idt/noiseless/IDTwithSeed" + str(seed)

        plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
        plt.close()
        
    #noise_idx = np.random.random(Y_train.shape)
    #Y_train_with_noise = Y_train.copy()
    #Y_train_with_noise[noise_idx<0.3] = 1 - Y_train_with_noise[noise_idx<0.3]

    
        noise_percentages = [0,0.5]
        noise_idx = np.random.random(Y_train.shape)
        
        plt.figure()
        for noise in noise_percentages:
            y_train_with_noise = Y_train.copy()
            #HACER ANDAR ESTA LINEA Y SALE.
            #y_train_with_noise[noise_idx<noise] = np.floor(y_train_with_noise[noise_idx<nois] - 1)#y_train_with_noise[noise_idx<noise] - 1
            y_train_with_noise[noise_idx<noise] = np.ones(y_train_with_noise[noise_idx<noise].shape) - y_train_with_noise[noise_idx<noise]
            trainScoresWithNoise = []
            testScoresWithNoise = []
            for i in max_nodes:
                clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train, Y_train)
                trainScoresWithNoise.append(clf.score(X_train,Y_train))
                testScoresWithNoise.append(clf.score(X_test,Y_test))
            plt.plot(max_nodes, trainScoresWithNoise,label='Train Score')
            plt.plot(max_nodes, testScoresWithNoise,label='Test Score error '+str(int(noise*100))+'%') 

        plt.ylabel("Score")
        plt.title("Sobreajuste en IDT con ruido en ") 
        plt.xlabel("Numero de nodos maximos")

        plt.legend()
        file_path = "../informe/idt/noise/IDTwithSeed" + str(seed)+"noise"+ str(int(noise*100))
        print file_path
        plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
        plt.close()


if __name__ == '__main__':
    main()


