import numpy as np
import cv2
import glob
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
def main(path):
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

def clasificador:


def Bagging(clasificador,n_estimators=10,max_samples=1,max_features=1,bootstrap=True,bootstrap_features=False,random_state=777): 
	return BaggingClassifier(clasificador,n_estimators=n_estimators,max_samples=max_samples, max_features=max_features,bootstrap=bootstrap,bootstrap_features=bootstrap_features,random_state=random_state)

def Boosting(n_estimators=100,random_state=777):
	return AdaBoostClassifier(n_estimators=n_estimators,random_state=random_state)

def RandomForest():



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
	if (len(sys.argv) < 4):
    print 'Usage: python src.py [path_to_images] '
else:
    path = (sys.argv[1])
    main(path)