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

import socket

while(1):
    '''
        send a request to Allocator so that he/she may allocate node
    '''
    # create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('', 8890))
    client.send('0')
    response = client.recv(4096)
    if response == 'no':
        #no request is left
        print 'no request is left exiting!'
        client.close()
        break
    '''
        Now execute and find the best n for the kmeans
    '''
    #query image
    image = cv2.imread("./database/"+response)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    '''
        Send the best n back to the Allocator
    '''
    client.close()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('', 8890))
    client.sendall(response+" "+str(n_cluster)+"\n")
    print 'response sent: ',response+" "+str(n_cluster)+"\n"
    client.close()
