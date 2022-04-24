#importing required libraries
import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/") 


from dataset import loadTestData, loadTrainData

from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
from skimage.feature import hog

k=300 #cluster size

def hog_extract():
    
   #load training and testing datasets
   trainData = loadTrainData()
   testData = loadTestData()

   #training features and hog images
   feature = [] #example: [[hog of seed1 top],[hog of seed1 left],.....,[hog of seed2 top],[...],[...],....]
   hog_im = []

   #testing features and hog images 
   feature_test  = []
   hog_test_im = []
   
   #perform hog on training seeds
   print("\nExtracting HOG features on Training datasets...")
   for i_training_data in range(len(trainData)):

       for i_view in range(len(trainData[i_training_data])):

            feature_set=[]
            hog_set=[]

            resized_img = cv2.resize(cv2.imread(trainData[i_training_data][i_view][0]), (40, 40))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(9, 9),
                              cells_per_block=(2, 2), visualize=True, multichannel=True)
    
            feature_set.append(fd)
            feature.append(feature_set)
            hog_set.append(hog_image)
            hog_im.append(hog_set)


   #perform hog on testing seeds   
   print("\nExtracting HOG features on Testing datasets...")
   for i_testing_data in range(len(testData)):

     for i_view in range(len(testData[i_testing_data])):

      feature_test_set=[]
      hog_test_set=[]
    
      resized_img = cv2.resize(cv2.imread(testData[i_testing_data][i_view][0]), (40, 40))
      fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(9, 9), 
                          cells_per_block=(2, 2), visualize=True, multichannel=True)
    
      feature_test_set.append(fd)
      feature_test.append(feature_test_set)
      hog_test_set.append(hog_image)
      hog_test_im.append(hog_test_set)
   
   return create_bof(feature, feature_test) #create bag of features for training and testing datasets


def create_bof(feature, feature_test):
  
  #stack descriptors of training seeds vertically
  descriptors_hog=feature[0]
  for descriptor in feature[1:]:
    if descriptor is None:
      pass
    else:
      descriptors_hog=np.vstack((descriptors_hog,descriptor)) 
  
  descriptors_hog_float=descriptors_hog.astype(float)
  
  #create codebook using features of training seeds
  print("\nGenerating HOG codebook...")
  voc_hog,variance=kmeans(descriptors_hog_float,k,1)
  
  #Creating bag of features and histogram for training seeds
  train_hog_features=np.zeros((len(feature),k),"float32")
  print("\nCreating bag of features for training datasets...")
  
  for i in range(len(feature)):
    if feature[i] is not None:

        # compare the descriptor with the vocabulary, to see which descriptor fall on which feature in the vocabulary
        words, distance = vq(feature[i],voc_hog) 
        
        for w in words:
            train_hog_features[i][w]+=1 # histogram accumulating, number of bins = length of vocabulary
            
  #Standardisation scaling
  stdslr=StandardScaler().fit(train_hog_features)
  train_hog_features=stdslr.transform(train_hog_features)
  print(train_hog_features.shape)
  
  #Creating bag of features and histogram of training image
  test_hog_features=np.zeros((len(feature_test),k),"float32")
  print("\nCreating bag of features for testing datasets...")
  
  for i in range(len(feature_test)):
    if feature_test[i] is not None:

        # compare the descriptor with the vocabulary, to see which descriptor fall on which feature in the vocabulary
        words, distance = vq(feature_test[i],voc_hog) 
        
        for w in words:
            test_hog_features[i][w]+=1 # histogram accumulating, number of bins = length of vocabulary
    
  #Standardisation scaling
  test_hog_features=stdslr.transform(test_hog_features)
  print(test_hog_features.shape)

  return train_hog_features, test_hog_features #bag of features
