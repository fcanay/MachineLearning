from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

    max_nodes = range(10,500,10)

    trainScores = []
    testScores = []
    for i in max_nodes:
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=i).fit(X_train, Y_train)
        trainScores.append(clf.score(X_train,Y_train))
        testScores.append(clf.score(X_test,Y_test))

    plt.figure(1)
    plt.title("Sobreajuste en IDT")
    plt.ylabel("Score")
    plt.xlabel("Numero de nodos maximos")
    plt.plot(max_nodes, trainScores,label='Train Score')
    plt.plot(max_nodes, testScores,label='Test Score') 

    plt.legend()
    plt.show()  






if __name__ == '__main__':
    main()


