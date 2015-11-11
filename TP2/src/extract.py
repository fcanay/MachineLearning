import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
import csv

def main(path,filename):
    print 'hola'
    X,y = load_images('/tmp/train/')
    print 'hola'
    with open('filename00.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(0,len(X)):
            print i
            spamwriter.writerow([y[i]] + list(X[i]))



def attributes_from(images):
    all_attributes = []
    for image in images:
        all_attributes.append(extract_attributes(image))
    return all_attributes

def extract_attributes(image):
    print "extrayendo atributo"
    X = histogramaColor(image)
    X1 = np.concatenate((X,histogramaByN(image)))
    X2 = np.concatenate((X,Patrones3x3ByN(image)))
    return np.concatenate((X2,Patrones2x2ByN(image)))
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
            pattern = str(oscuro(greyImage[i,j])) + str(oscuro(greyImage[i+1,j]))+ str(oscuro(greyImage[i,j+1]))+ str(oscuro(greyImage[i+1,j+1]))
            if pattern in patterns_dict:
                patterns_dict[pattern] += 1
            else:
                patterns_dict[pattern] = 1

    return patterns_dict.values()

def Patrones3x3ByN(image):
    greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns_dict = {'0000':0,'0001':0,'0010':0,'0100':0,'1000':0,'0011':0,'0101':0,'0110':0,'0111':0,'1001':0,'1010':0,'1011':0,'1100':0,'1101':0,'1110':0,'1111':0,}
    patterns_dict = [0] * 512
    for i in range(greyImage.shape[0]-2):
        for j in range(greyImage.shape[1]-2):
            pattern = oscuro(greyImage[i,j] + oscuro(greyImage[i+1,j] * 2 + oscuro(greyImage[i+2,j]) * 4 + oscuro(greyImage[i,j+1]) * 8 + oscuro(greyImage[i+1,j+1] * 16 + oscuro(greyImage[i+2,j+1]) * 32 + oscuro(greyImage[i,j+2]) * 64 + oscuro(greyImage[i+1,j+2] * 128 + oscuro(greyImage[i+2,j+2]) * 256
            patterns_dict[pattern] += 1

    return patterns_dict

def oscuro(gris):
    if gris > 127:
        return 1
    else:
        return 0

def load_images(path):
    imageFileNames = getImageFileNames(path)
    clases = get_clases_from(imageFileNames)
    images = []
    for myFile in imageFileNames:
        print 'Procesando' + myFile   
        images.append(cv2.imread(myFile))
    X = attributes_from(images)
    return X,clases

def getImageFileNames(path):
    imageFileNames = glob.glob(path +"*00.jpg")
    print imageFileNames
    return imageFileNames

def get_clases_from(imageFileNames):
    clases = []
    for fileName in imageFileNames:
        if fileName.find("dog") > -1 :
            clases.append("dog")
        if fileName.find("cat") > -1 :
            clases.append("cat")
    return clases

print 'hello'
if len(sys.argv) != 2 and False:
    print 'Usage: python src.py [path_to_images] '
else:
    path = True#(sys.argv[1])
    main(path,'filename')