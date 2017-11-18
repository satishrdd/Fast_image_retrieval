#required libraries

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import random
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--clusters", required = True, type = int,
# 	help = "# of clusters")
# args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib

#images for dataset few of them
image1 = cv2.imread("193000.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread("29002.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3 = cv2.imread("193039.jpg")
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)


#query image
imageq = cv2.imread("29012.jpg")
imageq = cv2.cvtColor(imageq, cv2.COLOR_BGR2RGB)


#list of all images

images=[image1,image2,image3,imageq]



#global values to find the random shifts to make

fxmin=9999
fymin=9999
fzmin=9999
fxmax=0
fymax=0
fzmax=0
fmaxl=0
# show our image

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

    xmax=0
    xmin=256
    ymin=256
    ymax=0
    zmin=256
    zmax=0

    #finding the border points of the image represented in cluster
    for points in ImgClt[imageCount].cluster_centers_:
    	xmax = max(xmax,points[0])
    	xmin = min(xmin,points[0])
    	ymax = max(ymax,points[1])
    	ymin = min(ymin,points[1])
    	zmax = max(zmax,points[2])
    	zmin = min(zmin,points[2])
    fxmin = min(fxmin,xmin)
    fxmax = max(fxmax,xmax)
    fymin = min(fymin,ymin)
    fymax = max(fymax,ymax)
    fzmin = min(fzmin,zmin)
    fzmax = max(fzmax,zmax)
    print xmax,xmin


    #finding the approximate diameter
    delta = int(max(xmax-xmin,max(ymax-ymin,zmax-zmin)))

    labelc = [0]*(n_cluster)
    for i in ImgClt[imageCount].labels_:
    	labelc[i]+=1
    flabels.append(labelc)

    maxl = max(xmax,max(ymax,zmax))
    fmaxl = max(fmaxl,maxl)
    imageCount+=1

#finding the unique random shift after running through all images

xshift = int(random.uniform(0,int(fxmin)))
yshift = int(random.uniform(0,int(fymin)))
zshift = int(random.uniform(0,int(fzmin)))
y=0
for image in images:
	grid=[[[0]*(10*int(fmaxl)+1)]*(10*int(fmaxl)+1)]*(10*int(fmaxl)+1)
	sidelen=1
	ve = []
	#embedding using various side lengths
	while sidelen<=fmaxl:
		for i in range(xshift,int(fxmax)+(sidelen-(int(fxmax)-xshift)%sidelen)%sidelen,sidelen):
			#print i
			for j in range(yshift,int(fymax)+(sidelen-(int(fymax)-yshift)%sidelen)%sidelen,sidelen):
				for k in range(zshift,int(fzmax)+(sidelen-(int(fzmax)-zshift)%sidelen)%sidelen,sidelen):
					l=0
					for points in ImgClt[y].cluster_centers_:
						if points[0]>=i and points[0]<i+1 and points[1]>=j and points[1]<j+1 and points[2]>=k and points[2]<k+1:
							grid[i][j][k] += flabels[y][l]
						l+=1
					ve.append(grid[i][j][k]*sidelen)
		sidelen*=2


	velist.append(ve)
	y+=1

res=99999999999999999
pos=-1

for i in velist:
	#print len(i)
	#print the f(p) - f(q) for each image for comparison
	print np.sum(np.absolute(np.subtract(np.array(i),np.array(velist[3]))))
