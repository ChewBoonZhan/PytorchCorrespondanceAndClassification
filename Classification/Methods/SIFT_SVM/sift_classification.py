
import os
import sys

import numpy as np
import cv2
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from dataset_sift import loadTestData,loadTrainData
from save_results_csv import save_results_csv
from save_results_image import save_results_image

sift = cv2.SIFT_create()

trainData = loadTrainData()
testData = loadTestData()

k=1000

def sift_extract():

    #shuffle training seeds
    np.random.shuffle(trainData) 
    
    #unzip data
    image_paths_train, train_labels = zip(*trainData) #y -> labels 1 = good 0 = bad
    image_paths_test, test_labels = zip(*testData)

    #SIFT + create bag of features for training and testing seeds
    train_list, train_features, voc, stdslr = sift_bof_train(image_paths_train)
    test_list, test_features = sift_bof_test(image_paths_test, voc, stdslr)

    #train and test svm model
    pred_labels, true_classes_sift, predict_classes_sift = svm(train_features, train_labels, test_features, test_labels)
    
    #evaluate results
    evaluate_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test)


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
    
    #standardize scaling
    test_features=stdslr.transform(test_features)

    return test_list, test_features


def svm(train_features, train_labels, test_features, test_labels):

    #train classification model
    print("\nTraining the SVM model using bag of features...")
    clf=LinearSVC(max_iter=80000)
    clf.fit(train_features,np.array(train_labels)) 
    
    #run on the trained model
    print("\nRunning testing seeds on the trained SVM model...")
    pred_labels = clf.predict(test_features)# predict classes of seed based on features

    #assign the actual name of seed classes instead of 0 and 1
    true_classes_sift=[]
    for i in test_labels:
        if i==1:
           true_classes_sift.append("GoodSeed")
        else:
           true_classes_sift.append("BadSeed")

    predict_classes_sift=[]
    for i in pred_labels:
        if i==1:
           predict_classes_sift.append("GoodSeed")
        else:
           predict_classes_sift.append("BadSeed")

    return pred_labels, true_classes_sift, predict_classes_sift


def evaluate_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test):

    print("\nSIFT")

    print("\nClassification Report")
    print(classification_report(test_labels, pred_labels, target_names = ['Bad Seeds','Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(test_labels, pred_labels, labels=range(2)))

    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_SIFT'
    path_csv_sift = os.path.join(path, 'classification_results.csv')

    false_score_good_sift, false_score_bad_sift, true_score_good_sift, true_score_bad_sift, total_bad_seeds_sift, total_good_seeds_sift = save_results_csv(path, path_csv_sift, true_classes_sift, predict_classes_sift, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds_sift)
    print("No. of Bad seeds detected correctly: ", true_score_bad_sift)
    print("No. of Bad seeds detected wrongly: ", false_score_bad_sift)

    print("\nTotal good testing seeds: ",total_good_seeds_sift)
    print("No. of Good seeds detected correctly: ", true_score_good_sift)
    print("No. of Good seeds detected wrongly: ", false_score_good_sift)

    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_SIFT/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_SIFT/Good_seeds/' 

    save_results_image(path_to_results_bad_seeds,path_to_results_good_seeds, pred_labels, image_paths_test)


