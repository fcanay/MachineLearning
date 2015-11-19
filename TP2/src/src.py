import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
import csv
from sklearn.ensemble import BaggingClassifier

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

	batchs = ['histogramaByN','histogramaColor','patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9']
	#batchs = ['patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9']
	
	X = []
	y = []
	load_batch(y,path,'clases',filename) 
	y = [j for i in y for j in i]
	for batch in batchs:
		load_batch(X,path,batch,filename)
	
	#X,y = load_images('/tmp/train/')
	est = [RandomForest(),Boosting(),Gradient(),SVM()]
	clf = VotingClassifier(estimators= est)
	#scores = cross_validation.cross_val_score(clf, X, y, cv=5)
	#print scores
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.2,random_state=777)
	#print clf.sub_score(X_test,Y_test)
	for name,estim in est:
		estim.fit(X_train,Y_train)
		print name
		print estim.score(X_test,Y_test)
	
	clf.fit(X_train,Y_train)
	print clf.score(X_test,Y_test)

	return


def load_batch(X,path,batch,filename):
	with open(path+filename+'_'+batch+'.csv', 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
		i = 0
		if X != []:
			for row in spamreader:
				X[i] += row
				i +=1
		else:	
			for row in spamreader:
				X.append(row)

class VoterClassifier(ClassifierMixin):
	def __init__(self):
		ClassifierMixin.__init__(self)
		self.clasificadores = [RandomForest(),Boosting(),Gradient(),SVM()]#	,Bagging()]

	def fit(self,X,y):
		for clasificador in self.clasificadores:
			clasificador.fit(X,y)

	def predict(self,X):
		res = []
		for clasificador in self.clasificadores:
			res.append(clasificador.predict(X))
		return mode(res)[0][0]

	def predict_proba(self, X):
		self.predictions_ = list()
		for classifier in self.classifiers:
			self.predictions_.append(classifier.predict_proba(X))
		return np.mean(self.predictions_, axis=0)

	def sub_score(self,X,y):
		res = []
		for clasificador in self.clasificadores:
			res.append(clasificador.score(X,y))
		return random_state


#Usar KNN
def Bagging(n_estimators=10,max_samples=1,max_features=1,bootstrap=True,bootstrap_features=False,random_state=777): 
	tuned_parameters = [{'n_estimators': [5,10,15] ,'max_samples':[0.7,1],'max_features':[0.7,1]}]
	return ('Bagging',GridSearchCV(BaggingClassifier(KNeighborsClassifier()), tuned_parameters, cv=2))

def Boosting(n_estimators=100,random_state=777):
	tuned_parameters = [{'n_estimators':[30,50,75],'learning_rate':[0.25,0.5,0.75,1]}]
	return ('Boosting',GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5))


def RandomForest(n_estimators=10,random_state=777):
	tuned_parameters = [{'n_estimators':[5,10,15],'max_features':['sqrt','log2',None],'bootstrap':[True,False]}]
	return ('RondomForest',GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5))
	return RandomForestClassifier()

def Gradient():
	tuned_parameters = [{'loss': ['deviance', 'exponential'],'n_estimators':[75,100,150] ,'learning_rate':[0.05,0.1,0.2]}]
	return ('Gradient',GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5))

def SVM():
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	return ('SVM',GridSearchCV(SVC(), tuned_parameters, cv=5))

def plotGridSearch():
	pass



if __name__ == '__main__':
	if len(sys.argv) != 3 and False:
		print 'Usage: python src.py [path_to_images] [batch_name]'
	else:
		path = (sys.argv[1])
		filename = (sys.argv[2])
		main(path,filename)
