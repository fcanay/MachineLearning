import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
import csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation
def main(path,filename):

	batchs = ['histogramaByN','histogramaColor','patrones2x2ByN','patronesCircularesByN_2_5','patronesCircularesByN_2_9','patronesCircularesByN_3_9','patronesCircularesByN_5_9','patronesCircularesByN_3_5']
	#batchs = ['patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9']
	percentil = 20
	X = []
	y = []
	lens = []
	load_batch(y,path,'clases',filename) 
	y = [j for i in y for j in i]
	for batch in batchs:
		load_batch(X,path,batch,filename)
		lens.append(len(X[0]))
	
	total = [lens[0]]
	for i in xrange(1,len(lens)):
		total.append(lens[i]-lens[i-1])
	print 'Cantidad de atributos por barch'
	print total
	sp = SelectPercentile(chi2,percentil)
	X_new = sp.fit_transform(X, y)
	sup = sp.get_support(True)
	#print sup
	res = [0]* len(batchs)
	for i in sup:
		for j in xrange(0,len(lens)):
			if i <= lens[j]:
				res[j] +=1
				break
	porcentajes = []
	for i in xrange(0,len(lens)):
		porcentajes.append((1.0*res[i])/total[i])
	print 'Cantidad de variables seleccionas en el'+str(percentil)+'percentil univariado'
	print res

	print 'Porcentaje de variables seleccionas en el'+str(percentil)+'percentil univariado'
	print porcentajes
	
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
	fi = clf.feature_importances_

	res2 = [0]* len(batchs)
	for i in xrange(0,len(fi)):
		for j in xrange(0,len(lens)):
			if i <= lens[j]:
				res2[j] += fi[i]
				break
	print 'Importancia porcentual acumulada de la seleccion multivariada'
	print res2
	porcentajes2 = []
	for i in xrange(0,len(lens)):
		porcentajes2.append((1.0*res2[i])/total[i])

	print 'Importancia porcentual promedio por variable de la seleccion multivariada'
	print porcentajes2


def load_batch(X,path,batch,filename):
	with open(path+filename+'_'+batch+'.csv', 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_NONNUMERIC)
		i = 0
		if X != []:
			for row in spamreader:
				X[i] += row
				i +=1
		else:	
			for row in spamreader:
				X.append(row)


if __name__ == '__main__':
	if len(sys.argv) != 3 and False:
		print 'Usage: python src.py [path_to_images] [batch_name]'
	else:
		path = (sys.argv[1])
		filename = (sys.argv[2])
		main(path,filename)