import numpy as np
import cv2
from statistics import mode
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
def main():
	img = cv2.imread('/tmp/perrogato/train/dog.999.jpg',0) 
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
        MySuperClass.__init__(self)
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
	tuned_parameters = []
	return GridSearchCV(BaggingClassifier(KNeighborsClassifier()), tuned_parameters, cv=5,scoring='%s_weighted' % score)

def Boosting(n_estimators=100,random_state=777):
	tuned_parameters = []
	return GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5,scoring='%s_weighted' % score)


def RandomForest(n_estimators=10,random_state=777):
	tuned_parameters = []
	return GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,scoring='%s_weighted' % score)

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


if __name__ == '__main__':
	main()