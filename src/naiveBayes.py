# coding= UTF-8

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

stop_words = [u'un',u'una',u'unas',u'unos',u'uno',u'sobre',u'todo',u'también',u'tras',u'otro',u'algún',u'alguno',u'alguna',u'algunos',u'algunas',u'ser',u'es',u'soy',u'eres',u'somos',u'sois',u'estoy',u'esta',u'estamos',u'estais',u'estan',u'como',u'en',u'para',u'atras',u'porque',u'porqué',u'estado',u'estaba',u'ante',u'antes',u'siendo',u'ambos',u'pero',u'por',u'poder',u'puede',u'puedo',u'podemos',u'podeis',u'pueden',u'fui',u'fue',u'fuimos',u'fueron',u'hacer',u'hago',u'hace',u'hacemos',u'haceis',u'hacen',u'cada',u'fin',u'incluso',u'primero',u'desde',u'conseguir',u'consigo',u'consigue',u'consigues',u'conseguimos',u'consiguen',u'ir',u'voy',u'va',u'vamos',u'vais',u'van',u'vaya',u'gueno',u'ha',u'tener',u'tengo',u'tiene',u'tenemos',u'teneis',u'tienen',u'el',u'la',u'lo',u'las',u'los',u'su',u'aqui',u'mio',u'tuyo',u'ellos',u'ellas',u'nos',u'nosotros',u'vosotros',u'vosotras',u'si',u'dentro',u'solo',u'solamente',u'saber',u'sabes',u'sabe',u'sabemos',u'sabeis',u'saben',u'ultimo',u'largo',u'bastante',u'haces',u'muchos',u'aquellos',u'aquellas',u'sus',u'entonces',u'tiempo',u'verdad',u'verdadero',u'verdadera',u'cierto',u'ciertos',u'cierta',u'ciertas',u'intentar',u'intento',u'intenta',u'intentas',u'intentamos',u'intentais',u'intentan',u'dos',u'bajo',u'arriba',u'encima',u'usar',u'uso',u'usas',u'usa',u'usamos',u'usais',u'usan',u'emplear',u'empleo',u'empleas',u'emplean',u'ampleamos',u'empleais',u'valor',u'muy',u'era',u'eras',u'eramos',u'eran',u'modo',u'bien',u'cual',u'cuando',u'donde',u'mientras',u'quien',u'con',u'entre',u'sin',u'trabajo',u'trabajar',u'trabajas',u'trabaja',u'trabajamos',u'trabajais',u'trabajan',u'podria',u'podrias',u'podriamos',u'podrian',u'podriais',u'yo',u'aquel']

def main():

    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               non_negative=True,stop_words=stop_words)
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
    #print(conf_matrix_normalized)

    plot_confusion_matrix(conf_matrix_normalized)
    #plt.matshow(conf_matrix)
    plt.show()

def plot_confusion_matrix(cm):
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(len(clases_unicode))
    plt.xticks(tick_marks, clases_unicode,rotation=-45)
    plt.yticks(tick_marks, clases_unicode)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
