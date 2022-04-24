
import os
import sys

import numpy as np
import cv2
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from dataset_sift import loadTestData,loadTrainData


sift = cv2.SIFT_create()

k=1000 #cluster size

def sift_extract():

    trainData = loadTrainData()
    testData = loadTestData()

    #shuffle training data
    np.random.shuffle(trainData) 
    
    #unzip data
    image_paths_train, train_labels = zip(*trainData)
    image_paths_test, test_labels = zip(*testData)

    train_list, train_features, voc, stdslr = sift_bof_train(image_paths_train)
    test_list, test_features = sift_bof_test(image_paths_test, voc, stdslr)
    
    return train_features, test_features, train_labels, test_labels, image_paths_test


def sift_bof_train(image_paths_train):

    
    train_list=[]

    #SIFT feature extraction
    print("\nSIFT feature extraction on training seeds...")
    for image_path_train in image_paths_train: #go through each seed
    
        img = cv2.imread(image_path_train)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #get image based on path
        keypoint, descriptor = sift.detectAndCompute(img, None)
        train_list.append((image_path_train,descriptor)) 

    #Generate Codebook using K-means
    descriptors=train_list[0][1]
    for image_path,descriptor in train_list[1:]:
       if descriptor is None:
         pass
       else:
         descriptors=np.vstack((descriptors,descriptor)) #stack descriptors of each image vertically

    descriptors_float=descriptors.astype(float)

    print("\nGenerating codebook...")
    voc,variance=kmeans(descriptors_float,k,1)
    
    #Creating bag of features and histogram of training image
    train_features=np.zeros((len(image_paths_train),k),"float32")
    print("\nCreating bag of features for training seeds...")

    for i in range(len(image_paths_train)):
        if train_list[i][1] is not None:

           # compare the descriptor with the vocabulary, to see which descriptor fall on which feature in the vocabulary
           words, distance = vq(train_list[i][1],voc) 
        
           for w in words:
               train_features[i][w]+=1 # histogram accumulating, number of bins = length of vocabulary

    #Standardisation scaling
    stdslr=StandardScaler().fit(train_features)
    train_features=stdslr.transform(train_features)

    return train_list, train_features, voc, stdslr



def sift_bof_test(image_paths_test, voc, stdslr):

    test_list=[]
    
    #SIFT feature extraction
    print("\nSIFT feature extraction on testing seeds...")
    for image_path_test in image_paths_test: #go through each seed
    
        img = cv2.imread(image_path_test)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #get image based on path
        keypoint, descriptor = sift.detectAndCompute(img, None)
        test_list.append((image_path_test,descriptor)) 


    #create bag of features
    test_features=np.zeros((len(image_paths_test),k),"float32")
    print("\nCreating bag of features for testing seeds...")

    for i in range(len(image_paths_test)):
        if test_list[i][1] is not None:
            words,distance=vq(test_list[i][1],voc)
            
            for w in words:
                test_features[i][w]+=1 

    test_features=stdslr.transform(test_features)

    return test_list, test_features 

