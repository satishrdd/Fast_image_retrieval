#required libraries

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import random
import numpy as np
 
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


#these are values to know the weight of each cluster in the cluster representation of image
flabels =[]


#this contains the vector generated as discussed in paper after embedding

velist=[]

for image in images:
	plt.figure()
	plt.axis("off")
	plt.imshow(image)

	image = image.reshape((image.shape[0] * image.shape[1], 3))

	#clusteing using kmeans first by 36 clusters and then by 9 clusters(mentioned in paper for 8.8 avg)

	clt1 = KMeans(n_clusters = 36)
	clt1.fit(image)
	clt= KMeans(n_clusters=9)
	clt.fit(clt1.cluster_centers_)

	print clt.cluster_centers_
	print len(clt.labels_)

	xmax=0
	xmin=256
	ymin=256
	ymax=0
	zmin=256
	zmax=0
	
	#finding the border points of the image represented in cluster
	for points in clt.cluster_centers_:
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
	


	labelc = [0]*(9)
	for i in clt.labels_:
		labelc[i]+=1
	flabels.append(labelc)
	
	maxl = max(xmax,max(ymax,zmax))
	fmaxl = max(fmaxl,maxl)


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
					for points in clt.cluster_centers_:
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




