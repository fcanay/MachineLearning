import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
from sklearn.ensemble import BaggingClassifier

from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
def main(path):
	X,y = load_images('/tmp/train/')
	clf = VoterClassifier()
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.5,random_state=777)
	clf.fit(X_train,Y_train)
	print clf.score(X_test,Y_test)
	print clf.sub_score(X_test,Y_test)
	return


	img = cv2.imread('/tmp/train/dog.999.jpg') 
	print extract_attributes(img)
	


	return




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

	# d = [0,0,0,0,0]
	# for i in range(equ.shape[0]-1):
	# 	for j in range(equ.shape[1]-1):
	# 		p = oscuro(equ[i,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j+1])
	# 		d[p] += 1
			
	# print d

class VoterClassifier(ClassifierMixin):
	def __init__(self):
		ClassifierMixin.__init__(self)
		self.clasificadores = [RandomForest(),Boosting()]#,Bagging()]

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
		return res


#Usar KNN
def Bagging(n_estimators=10,max_samples=1,max_features=1,bootstrap=True,bootstrap_features=False,random_state=777): 
	tuned_parameters = [{'n_estimators': [5,10,15] ,'max_samples':[0.3,0.5,0.7,1],'max_features':[0.5,0.7,1],'bootstrap':[True,False],'bootstrap_features':[True,False]}]
	return GridSearchCV(BaggingClassifier(KNeighborsClassifier(n_neighbors=2)), tuned_parameters, cv=5)

def Boosting(n_estimators=100,random_state=777):
	tuned_parameters = [{'n_estimators':[30,50,75],'learning_rate':[0.25,0.5,0.75,1]}]
	return GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5)


def RandomForest(n_estimators=10,random_state=777):
	tuned_parameters = [{'n_estimators':[5,10,15],'max_features':['sqrt','log2',None],'bootstrap':[True,False]}]
	return GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)

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
		all_attributes.append(extract_attributes(image))
	return all_attributes

def extract_attributes(image):
	print "extrayendo atributo"
	X = histogramaColor(image)
	X1 = np.concatenate((X,histogramaByN(image)))
	return np.concatenate((X1,Patrones2x2ByN(image)))
	Patrones2x2Color(image)
	PatronesCircularByN(image)
	return []

def histogramaColor(image):
	b,g,r = cv2.split(image)
	b_hist,bins = np.histogram(b.flatten(),256,[0,256])
	g_hist,bins = np.histogram(g.flatten(),256,[0,256])
	r_hist,bins = np.histogram(r.flatten(),256,[0,256])
	return np.concatenate((b_hist,g_hist,r_hist))

def histogramaByN(image):
	greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist,bins = np.histogram(greyImage.flatten(),256,[0,256])
	return hist

def Patrones2x2ByN(image):
	greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	patterns_dict = {'0000':0,'0001':0,'0010':0,'0100':0,'1000':0,'0011':0,'0101':0,'0110':0,'0111':0,'1001':0,'1010':0,'1011':0,'1100':0,'1101':0,'1110':0,'1111':0,}
	for i in range(greyImage.shape[0]-1):
		for j in range(greyImage.shape[1]-1):
			pattern = str(oscuro(greyImage[i,j])) + str(oscuro(greyImage[i+1,j]))+ str(oscuro(greyImage[i+1,j]))+ str(oscuro(greyImage[i+1,j+1]))
			if pattern in patterns_dict:
				patterns_dict[pattern] += 1
			else:
				patterns_dict[pattern] = 1

	return patterns_dict.values()


def load_images(path):
	imageFileNames = getImageFileNames(path)
	clases = get_clases_from(imageFileNames)
	images = []
	for myFile in imageFileNames:
		images.append(cv2.imread(myFile)) 
	X = attributes_from(images)
	return X,clases

def getImageFileNames(path):
	imageFileNames = glob.glob(path +"*000.jpg")
	return imageFileNames

def get_clases_from(imageFileNames):
	clases = []
	for fileName in imageFileNames:
		if fileName.find("dog") > -1 :
			clases.append("dog")
		if fileName.find("cat") > -1 :
			clases.append("cat")
	return clases


if __name__ == '__main__':
	if len(sys.argv) != 2 and False:
		print 'Usage: python src.py [path_to_images] '
	else:
		path = True#(sys.argv[1])
		main(path)
