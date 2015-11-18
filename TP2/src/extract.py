import numpy as np
import cv2
from scipy.stats import mode
import glob
import sys
import csv
import mahotas

def main(path,filename):
    print 'hola'
    load_images(path,filename)#'/tmp/train/')
    print 'hola'

def writeCSV(extractorName,batchName,attr):
    with open(batchName +'_'+ extractorName + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in xrange(0,len(attr)):
            spamwriter.writerow(list(attr[i]))



def attributes_from(images,fileName):
    attr_hist_Color = []
    attr_hist_ByN = []
    attr_Patrones2x2_ByN = []
    attr_Patrones3x3_ByN = []

    attr_PatronesCirc_ByN_3_9 = []
    attr_PatronesCirc_ByN_2_9 = []
    attr_PatronesCirc_ByN_2_5 = []
    attr_PatronesCirc_ByN_3_5 = []
    attr_PatronesCirc_ByN_5_9 = []

    i = 1
    for image in images:
        print "extrayendo atributos imagen " + str(i)
        i += 1
        attr_hist_Color.append(histogramaByN(image))
        #attr_Patrones2x2_Color.append(Patrones2x2Color(image))
        #attr_Patrones3x3_Color.append()
        #attr_PatronesCirc_Color.append(PatronesCircular(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        attr_hist_ByN.append(histogramaByN(image))
        attr_Patrones2x2_ByN.append(Patrones2x2ByN(image))
        attr_Patrones3x3_ByN.append(Patrones3x3ByN(image))

        attr_PatronesCirc_ByN_3_9.append(PatronesCircularByN(image,3,9))
        attr_PatronesCirc_ByN_2_9.append(PatronesCircularByN(image,2,9))
        attr_PatronesCirc_ByN_2_5.append(PatronesCircularByN(image,2,5))
        attr_PatronesCirc_ByN_3_5.append(PatronesCircularByN(image,3,5))
        attr_PatronesCirc_ByN_5_9.append(PatronesCircularByN(image,5,9))
    writeCSV('histogramaColor',fileName,attr_hist_Color)
    writeCSV('histogramaByN',fileName,attr_hist_ByN)
    writeCSV('patrones2x2ByN',fileName,attr_Patrones2x2_ByN)
    writeCSV('patrones3x3ByN',fileName,attr_Patrones3x3_ByN)
    writeCSV('patronesCirculaesByN_3_9',fileName,attr_PatronesCirc_ByN_3_9)
    writeCSV('patronesCirculaesByN_2_9',fileName,attr_PatronesCirc_ByN_2_9)
    writeCSV('patronesCirculaesByN_2_5',fileName,attr_PatronesCirc_ByN_2_5)
    writeCSV('patronesCirculaesByN_3_5',fileName,attr_PatronesCirc_ByN_3_5)
    writeCSV('patronesCirculaesByN_5_9',fileName,attr_PatronesCirc_ByN_5_9)

def histogramaColor(image):
    b,g,r = cv2.split(image)
    b_hist,bins = np.histogram(b.flatten(),256,[0,256])
    g_hist,bins = np.histogram(g.flatten(),256,[0,256])
    r_hist,bins = np.histogram(r.flatten(),256,[0,256])
    return np.concatenate((b_hist,g_hist,r_hist))

def histogramaByN(image):

    hist,bins = np.histogram(image.flatten(),256,[0,256])
    return hist

def Patrones2x2ByN(greyImage):
    #greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns_dict = {'0000':0,'0001':0,'0010':0,'0100':0,'1000':0,'0011':0,'0101':0,'0110':0,'0111':0,'1001':0,'1010':0,'1011':0,'1100':0,'1101':0,'1110':0,'1111':0,}
    for i in xrange(greyImage.shape[0]-1):
        for j in xrange(greyImage.shape[1]-1):
            pattern = str(oscuro(greyImage[i,j])) + str(oscuro(greyImage[i+1,j]))+ str(oscuro(greyImage[i,j+1]))+ str(oscuro(greyImage[i+1,j+1]))
            if pattern in patterns_dict:
                patterns_dict[pattern] += 1
            else:
                patterns_dict[pattern] = 1

    return patterns_dict.values()

def Patrones3x3ByN(greyImage):
    #greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns_dict = [0] * 512
    for i in xrange(greyImage.shape[0]-2):
        for j in xrange(greyImage.shape[1]-2):
            pattern = 0
            for x in xrange(0,3):
                for y in xrange(0,3):
                    pattern += oscuro(greyImage[i+x,j+y]) * (2**(y+(3*x)))
            #pattern = oscuro(greyImage[i,j]) + oscuro(greyImage[i+1,j]) * 2 + oscuro(greyImage[i+2,j]) * 4 + oscuro(greyImage[i,j+1]) * 8 + oscuro(greyImage[i+1,j+1] * 16 + oscuro(greyImage[i+2,j+1]) * 32 + oscuro(greyImage[i,j+2]) * 64 + oscuro(greyImage[i+1,j+2] * 128 + oscuro(greyImage[i+2,j+2]) * 256
            patterns_dict[pattern] += 1

    return patterns_dict

def PatronesCircularByN(image,radius,points):
	return mahotas.features.lbp(image, radius, points, ignore_zeros=False)

def surf_extraccion(image,number_of_points,hessian=400):
    surf = cv2.SURF(hessian)
    surf.upright = True
    kp= surf.detect(img,None)
    #kp, des= surf.detectAndCompute(img,None)

def oscuro(gris):
    if gris > 127:
        return 1
    else:
        return 0

def load_images(path,fileName):
    imageFileNames = getImageFileNames(path)
    writeCSV('clases',fileName,get_clases_from(imageFileNames))
    images = []
    for myFile in imageFileNames:
        print 'Procesando' + myFile
        images.append(cv2.imread(myFile))
    attributes_from(images,fileName)

def getImageFileNames(path):
    imageFileNames = glob.glob(path +"*0000.jpg")
    print imageFileNames
    return imageFileNames

def get_clases_from(imageFileNames):
    clases = []
    for fileName in imageFileNames:
        if fileName.find("dog") > -1 :
            clases.append("1")
        if fileName.find("cat") > -1 :
            clases.append("0")
    return clases


print 'hello'
if len(sys.argv) != 3:# and False:
    print 'Usage: python src.py [path_to_images] [batch_name] '
else:
    path = sys.argv[1] #'/home/enano/train/train/'#(sys.argv[1])
    batchName = sys.argv[2]
    main(path,batchName)
