import numpy as np
import urllib
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import random as rn
from sklearn.feature_extraction.text import HashingVectorizer
import csv

clases = ['Cultura',
 'Deportes',
 'Econom\xc3\xada',
 'Espect\xc3\xa1culos',    
 'Internacionales',
 'Pol\xc3\xadtica',
 'Seguridad',
 'Sociedad',
 'Tecnolog\xc3\xada']

clases_unicode = ['Cultura',
 'Deportes',
 'Economida',
 'Espectaculos',    
 'Internacionales',
 'Politica',
 'Seguridad',
 'Sociedad',
 'Tecnologia']

def main():

    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               non_negative=True)
    #PARA AGREGAR stopwords usar el kward stop_words y pasar lista de stop words

    y = []
    text = []

    with open('notas_ln.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            y.append(row[2])
            text.append(row[3])
       

    y.pop(0)
    text.pop(0)

    le = preprocessing.LabelEncoder()
    le.fit(clases)
    y = le.transform(y)

    X = vectorizer.transform(text)

    #X = np.c_[X,text]

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    
    clf = MultinomialNB(alpha=0.01)
    clf.fit(X_train,Y_train)

    conf_matrix = metrics.confusion_matrix(Y_test,clf.predict(X_test))
    print(conf_matrix)
    #plt.figure()
    plot_confusion_matrix(conf_matrix)

    
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    #plt.figure()
    plot_confusion_matrix(conf_matrix_normalized, title='Normalized confusion matrix')
    #plt.matshow(conf_matrix)
    plt.show()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(len(clases_unicode))
    plt.xticks(tick_marks, clases_unicode,rotation=-45)
    plt.yticks(tick_marks, clases_unicode)
    #plt.tight_layout()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
