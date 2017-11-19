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

fil = open("databaseSequential.txt","w")
imageCount = 0
for image in images:
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    '''
    Using Silhoutte score to get the best number of clusters
    can make this distributed for each image can return the best n_cluster!
    '''
    n_cluster = 8
    Prevscore = 2
    count = 0
    ElbowPoint = -1
    for centers in range(2,50):
        clt = KMeans(n_clusters = centers)
        labelsP = clt.fit_predict(image)
        silScore = silhouette_score(image,labelsP)
        if Prevscore - silScore < .018:
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

    print imageCount,'/',len(images)
    fil.write(imageFileNames[imageCount]+" "+str(n_cluster)+"\n")
    imageCount+=1

images,imageFileNames = load_images("./query/")

fil = open("querySequential.txt","w")
imageCount = 0
for image in images:
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    '''
    Using Silhoutte score to get the best number of clusters
    can make this distributed for each image can return the best n_cluster!
    '''
    n_cluster = 8
    Prevscore = 2
    count = 0
    ElbowPoint = -1
    for centers in range(2,50):
        clt = KMeans(n_clusters = centers)
        labelsP = clt.fit_predict(image)
        silScore = silhouette_score(image,labelsP)
        if Prevscore - silScore < .018:
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

    print imageCount,'/',len(images)
    fil.write(imageFileNames[imageCount]+" "+str(n_cluster)+"\n")
    imageCount+=1
