'''
Idea create a queue of requests and assign each node one request as and when it happens
Assuming each image on each of the requesters server
'''

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
    return imageFileNames

imageFileNames = load_images("./database/")

fil = open("DistributedDatabase.txt","w")
fil_existing_database = open("database.txt","r")

'''
Array to store for which files kmeans has already happend
'''

fileNameDict = []

for line in fil_existing_database:
    fileNameDict.append(line.split(' ')[0])

'''
introduce a new thread here for server and listerner
'''

notYetKmeaned = []
for fileName in imageFileNames:
    if fileName not in fileNameDict:
        notYetKmeaned.append(fileName)

'''
Listener And Allocator
NOTE: Assumption No Failures
'''

'''
    Simple socket server using threads
'''

import socket
import sys

HOST = ''
PORT = 8890

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket created'

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()

print 'Socket bind complete'

#Start listening on socket
s.listen(10)
print 'Socket now listening'

intialLengthOfNotYetKmeaned = len(notYetKmeaned)

recieved = 0

while(len(notYetKmeaned)>0 or recieved<intialLengthOfNotYetKmeaned):
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print 'Connected with ' + addr[0] + ':' + str(addr[1])
    data = conn.recv(1024)
    print data
    if(len(data) == 1):
        #request recieved
        toAllocate = "no"
        if(len(notYetKmeaned) != 0):
            toAllocate = notYetKmeaned.pop()
        conn.sendall(toAllocate)
    else:
        '''
        If Answer comes write to file
        '''
        recieved+=1
        print 'response recieved: ',data,'count',recieved,'intialLengthOfNotYetKmeaned',intialLengthOfNotYetKmeaned
        fil.write(data+'\n')
