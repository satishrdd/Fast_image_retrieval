from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import random
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.image as mpimg
import os
import math

def formula(cubeX,cubeY,cubeZ,sidelen,xshift,yshift,zshift):
    xc = math.ceil(((xshift%sidelen) + cubeX + 1)*1.0/sidelen)
    yc = math.ceil(((yshift%sidelen) + cubeY + 1)*1.0/sidelen)
    zc = math.ceil(((zshift%sidelen) + cubeZ + 1)*1.0/sidelen)
    yd = math.ceil((256+yshift%sidelen)*1.0/sidelen)
    zd = math.ceil((256+zshift%sidelen)*1.0/sidelen)
    xd = math.ceil((256+xshift%sidelen)*1.0/sidelen)
    offset = 0
    for side in range(0,int(math.log(sidelen,2))):
        yp = math.ceil((256+yshift%(2**side))*1.0/((2**side)))
        zp = math.ceil((256+zshift%(2**side))*1.0/(2**side))
        xp = math.ceil((256+xshift%(2**side))*1.0/(2**side))
        offset += xp*yp*zp
    #offset = int(math.log(sidelen,2))*xd*yd*zd
    return int(offset + (xc - 1)*yd*zd + (yc-1)*zd + zc - 1)

def load_images(folder):
    images = []
    imageFileNames = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            imageFileNames.append(filename)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images,imageFileNames

images,imageFileNames = load_images("./database/")

#query image
imageq = cv2.imread("query.jpg")
imageq = cv2.cvtColor(imageq, cv2.COLOR_BGR2RGB)
images.append(imageq)

#global values to find the random shifts to make

'''
store clt of every image in this:
'''
ImgClt = {}

#these are values to know the weight of each cluster in the cluster representation of image
flabels =[]


#this contains the vector generated as discussed in paper after embedding

velist=[]

imageCount = 0

for image in images:
    plt.figure()
    plt.axis("off")
    plt.imshow(image)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #clusteing using kmeans first by 36 clusters and then by 9 clusters(mentioned in paper for 8.8 avg)
    '''
    Using Silhoutte score to get the best number of clusters
    can make this distributed for each image can return the best n_cluster!
    '''
    n_cluster = 8
    Prevscore = 2
    count = 0
    ElbowPoint = -1
    for centers in range(2,50):
        ImgClt[imageCount] = KMeans(n_clusters = centers)
        labelsP = ImgClt[imageCount].fit_predict(image)
        silScore = silhouette_score(image,labelsP)
        print silScore, "center: ",centers
        if Prevscore - silScore < .02:
            count+=1
            if count == 1:
                ElbowPoint = centers - 1
            if count > 3:
                n_cluster = ElbowPoint
                break
        else:
            count = 0
        n_cluster = centers
        Prevscore = silScore

    print n_cluster
    ImgClt[imageCount] = KMeans(n_clusters = n_cluster)
    ImgClt[imageCount].fit(image)

    print ImgClt[imageCount].cluster_centers_
    print len(ImgClt[imageCount].labels_)

    #finding the approximate diameter

    labelc = [0]*(n_cluster)
    for i in ImgClt[imageCount].labels_:
    	labelc[i]+=1
    flabels.append(labelc)
    imageCount+=1

#finding the unique random shift after running through all images

xshift = int(random.uniform(0,255))
yshift = int(random.uniform(0,255))
zshift = int(random.uniform(0,255))

print "Done Creating Cube Embedding, moving on to finding number of points in each cube"

'''
try to store the values per id obtained for cubes above in this
'''

imageId = 0

velist = []

for image in images:
    sidelen = 1
    mappingPointsToCube = [0]*(256*256*256*9)
    while sidelen<=255:
        for points in ImgClt[imageId].cluster_centers_:
            [x,y,z] = points
            CubeX = int(x/sidelen)*sidelen + (sidelen-(xshift%sidelen))%sidelen
            CubeY = int(y/sidelen)*sidelen + (sidelen-(yshift%sidelen))%sidelen
            CubeZ = int(z/sidelen)*sidelen + (sidelen-(zshift%sidelen))%sidelen
            mappingPointsToCube[formula(CubeX,CubeY,CubeZ,sidelen,xshift,yshift,zshift)]+=(sidelen)
        sidelen*=2
    velist.append(mappingPointsToCube)

    imageId+=1

print "Embedding for each image complete.! check once for speed"

for i in range(0,len(velist)):
	#print len(i)
	#print the f(p) - f(q) for each image for comparison
	print "diff from query: ",np.sum(np.absolute(np.subtract(np.array(velist[i]),np.array(velist[len(velist)-1]))))

veq = velist.pop()
psdL1_mat = libpylshbox.psdlsh()
psdL1_mat.init_mat(velist, '', 2, 1, 1, 5)
result = psdL1_mat.query(veq, 2, 10)
indices, dists = result[0], result[1]
for i in range(len(indices)):
    print indices[i], '\t', dists[i]
