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
from sklearn import metrics
from sklearn.svm import SVC
import cPickle as pickle

def main(path,filename):

	batchsT = ['histogramaByN','histogramaColor','patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9','patronesCirculaesByN_3_9','patronesCirculaesByN_5_9','patronesCirculaesByN_3_5']
	batchsAux = ['histogramaByN','histogramaColor','patronesCirculaesByN_2_5','patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_9','patronesCirculaesByN_3_9','patronesCirculaesByN_5_9','patronesCirculaesByN_3_5','patronesCirculaesByN_6_12','patronesCirculaesByN_8_12']
	#batchs = ['patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_2_9']
	#batchs = ['patrones2x2ByN','patrones3x3ByN','patronesCirculaesByN_2_5','patronesCirculaesByN_3_5']
	#for batch in batchsAux:


	#print batch
	batchs = batchsAux
	#batchs.remove(batch)
	X = []
	y = []
	load_batch(y,path,'clases',filename) 
	y = [j for i in y for j in i]
	for batch in batchs:
		load_batch(X,path,batch,filename)
	
	#X,y = load_images('/tmp/train/')
	est = [RandomForest(),Boosting()]
	for i in xrange(0,15):
		est.append(Gradient(i))
	for i in xrange(0,4):
		est.append(SVM(i))

	#scores = cross_validation.cross_val_score(clf, X, y, cv=5)
	#print scores
	clf = VotingClassifier(estimators=est)

	clf.fit(X,y)
	pickle.dump( clf, open( "clf_grande.p", "wb" ) )
	return
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.2,random_state=777)
	#print clf.sub_score(X_test,Y_test)
	print 'start'
	conf_matrix = metrics.confusion_matrix(Y_test,clf.predict(X_test))
	print 'confution matrix'
	print conf_matrix
	return
	for name,estim in est:
		print name
		#estim.fit(X_train,Y_train)
		#print estim.score(X_test,Y_test)
		print cross_validation.cross_val_score(estim, X, y, cv=5,n_jobs=-1)
	print 'voter'
	print cross_validation.cross_val_score(clf, X, y, cv=5,n_jobs=-1)
	return
	#clf.fit(X_train,Y_train)
	print clf.score(X_test,Y_test)

	return


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

class VoterClassifier(ClassifierMixin):
	def __init__(self):
		ClassifierMixin.__init__(self)
		self.clasificadores = [RandomForest(),Boosting(),Gradient(),SVM(),SVM2()]#	,Bagging()]

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
	return ('Bagging',GridSearchCV(BaggingClassifier(KNeighborsClassifier()), tuned_parameters, cv=2,n_jobs=-1))

def Boosting(n_estimators=100,random_state=777):
	return ('Boosting',AdaBoostClassifier())
	tuned_parameters = [{'n_estimators':[30,50,75],'learning_rate':[0.25,0.5,0.75,1]}]
	return ('Boosting',GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5,n_jobs=-1))


def RandomForest(n_estimators=10,random_state=777):
	return ('RandomForest',RandomForestClassifier())
	tuned_parameters = [{'n_estimators':[5,10,15],'max_features':['sqrt','log2',None],'bootstrap':[True,False]}]
	return ('RandomForest',GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,n_jobs=-1))

def Gradient(i=0):
	if i==0:
		return ('Gradient'+str(i),GradientBoostingClassifier(random_state=i))
	elif i==1:
		return ('Gradient'+str(i),GradientBoostingClassifier(loss='exponential',random_state=i))
	elif i==2:
		return ('Gradient'+str(i),GradientBoostingClassifier(loss='exponential',learning_rate=0.2,random_state=i))
	elif i==3:
		return ('Gradient'+str(i),GradientBoostingClassifier(learning_rate=0.2,random_state=i))
	elif i==4:
		return ('Gradient'+str(i),GradientBoostingClassifier(n_estimators=80,learning_rate=0.2,random_state=i))
	elif i==5:
		return ('Gradient'+str(i),GradientBoostingClassifier(n_estimators=120,learning_rate=0.2,random_state=i))
	elif i==6:
		return ('Gradient'+str(i),GradientBoostingClassifier(max_depth=2,loss='exponential',learning_rate=0.2,random_state=i))
	elif i==7:
		return ('Gradient'+str(i),GradientBoostingClassifier(max_depth=4,loss='exponential',learning_rate=0.2,random_state=i))
	
	else:
		return ('Gradient'+str(i),GradientBoostingClassifier(random_state=i))
	tuned_parameters = [{'loss': ['deviance', 'exponential'],'n_estimators':[75,100,150] ,'learning_rate':[0.05,0.1,0.2]}]
	return ('Gradient',GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5,n_jobs=-1))
	return ('Gradient',GradientBoostingClassifier())

def SVM(i=0):
	if i==0:
		return ('SVM'+str(i),SVC(kernel='linear',shrinking=False,random_state=i))
	elif i==1:
		return ('SVM'+str(i),SVC(kernel='linear',random_state=i))
	elif i==2:
		return ('SVM'+str(i),SVC(kernel='linear',C=100,random_state=i))
	elif i==3:
		return ('SVM'+str(i),SVC(kernel='linear',C=10,random_state=i))
	elif i==4:
		return ('SVM'+str(i),SVC(gamma=1e-3,random_state=i))
	elif i==5:
		return ('SVM'+str(i),SVC(gamma=1e-4,random_state=i))
	else:
		return ('SVM'+str(i),SVC(random_state=i))

	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	return ('SVM',GridSearchCV(SVC(), tuned_parameters, cv=5,n_jobs=-1))
	return ('SVM',SVC())

def SVM2():
	return ('SVM2',SVC(kernel='lineal'))

def plotGridSearch():
	pass



if __name__ == '__main__':
	if len(sys.argv) != 3 and False:
		print 'Usage: python src.py [path_to_images] [batch_name]'
	else:
		path = (sys.argv[1])
		filename = (sys.argv[2])
		main(path,filename)
