import numpy as np
import urllib
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random as rn
import csv

def main():
	X = []
	Y = []
	with open('notas_In.csv', 'rb') as csvfile:
	    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	    for row in spamreader:
	        X.append()
	        Y.append()

	clf = MultinomialNB()



if __name__ == '__main__':
    main()
