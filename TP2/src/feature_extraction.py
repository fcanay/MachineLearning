import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
import csv
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
def main(path,filename):

	batchs = ['histogramaByN','histogramaColor','patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9','patronesCirculaesByN_3_9','patronesCirculaesByN_5_9','patronesCirculaesByN_3_5']
	#batchs = ['patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9']
	
	X = []
	y = []
	load_batch(y,path,'clases',filename) 
	y = [j for i in y for j in i]
	for batch in batchs:
		load_batch(X,path,batch,filename)
	
	sp = SelectPercentile(chi2,50)
	X_new = sp.fit_transform(X, y)
	print X_new.shape
	print sp.get_support(True)




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