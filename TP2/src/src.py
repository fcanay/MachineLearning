import numpy as np
import cv2
from scipy.stats import mode
import glob
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
<<<<<<< HEAD
from sklearn.base import ClassifierMixin
def main(path):
	img = cv2.imread('/tmp/train/dog.999.jpg',0) 
	equ = cv2.equalizeHist(img)
	equ = img
	patterns_dict = {}
	for i in range(equ.shape[0]-1):
		for j in range(equ.shape[1]-1):
			pattern = str(oscuro(equ[i,j])) + str(oscuro(equ[i+1,j]))+ str(oscuro(equ[i+1,j]))+ str(oscuro(equ[i+1,j+1]))
			if pattern in patterns_dict:
				patterns_dict[pattern] += 1
			else:
				patterns_dict[pattern] = 1

	print patterns_dict

	d = [0,0,0,0,0]
	for i in range(equ.shape[0]-1):
		for j in range(equ.shape[1]-1):
			p = oscuro(equ[i,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j+1])
			d[p] += 1
			
	print d

class VoterClassifier(ClassifierMixin):
	def __init__(self):
		ClassifierMixin.__init__(self)
		self.clasificadores = [Bagging(),Boosting(),RandomForest()]

	def fit(X,y):
		for clasificador in self.clasificadores:
			clasificador.fit(X,y)

	def predict(X):
		res = []
		for clasificador in self.clasificadores:
			res.append(clasificador.predict(X))
		return mode(res)[0][0]

	def predict_proba(self, X):
		self.predictions_ = list()
		for classifier in self.classifiers:
			self.predictions_.append(classifier.predict_proba(X))
		return np.mean(self.predictions_, axis=0)


#Usar KNN
def Bagging(clasificador,n_estimators=10,max_samples=1,max_features=1,bootstrap=True,bootstrap_features=False,random_state=777): 
	tuned_parameters = [{'n_estimators': [5,10,15] ,'max_samples':[0.3,0.5,0.7,1],'max_features':[0.5,0.7,1],'bootstrap':[True,False],'bootstrap_features':[True,False]}]
	return GridSearchCV(BaggingClassifier(KNeighborsClassifier()), tuned_parameters, cv=5,scoring='%s_weighted' % score)

def Boosting(n_estimators=100,random_state=777):
	tuned_parameters = [{'n_estimators':[30,50,75],'learning_rate':[0.25,0.5,0.75,1]}]
	return GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5,scoring='%s_weighted' % score)


def RandomForest(n_estimators=10,random_state=777):
	tuned_parameters = [{'n_estimators':[],'max_features':[],'bootstrap':[True,False]}]
	return GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,scoring='%s_weighted' % score)

def plotGridSearch():
	pass
 # print("Best parameters set found on development set:")
 #    print()
 #    print(clf.best_params_)
 #    print()s
 #    print("Grid scores on development set:")
 #    print()
 #    for params, mean_score, scores in clf.grid_scores_:
 #        print("%0.3f (+/-%0.03f) for %r"
 #              % (mean_score, scores.std() * 2, params))
 #    print()


def oscuro(gris):
	if gris > 127:
		return 1
	else:
		return 0

def attributes_from(images):
	all_attributes = [] 
	for image in images:
		attributes = extractAttributes(image);
	return all_attributes

def extract_attributes(image):
	return []
	
def load_images(path):
	imageFileNames = glob.glob(path +"*.jpg")
	return imageFilesm

def get_clases_from(imageFileNames):
	clases = []
	for fileName in imageFileNames:
		if fileName.find("dog") > -1 :
			clases.Add("dog")
		if fileName.find("cat") > -1 :
			clases.Add("cat")
	return clases


if __name__ == '__main__':
	if (len(sys.argv) != 2):
    print 'Usage: python src.py [path_to_images] '
else:
    path = (sys.argv[1])
    main(path)
