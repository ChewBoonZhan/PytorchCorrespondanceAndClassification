from cgi import test
import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from dataset import *
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from save_results_csv import save_results_csv
from save_results_corr_image import save_results_corr_image

trainData = loadTrainData()
testData = loadTestData()

def extract_hog_feature():
    hog_train = []
    hog_test = []

    for i_training_data in range(len(trainData)):
        feature_set  = []
    
        for i_view in range(len(trainData[i_training_data])):
            resized_img = cv2.resize(cv2.imread(trainData[i_training_data][i_view][0]), (42, 42))
            fd, _ = hog(resized_img, orientations=9, pixels_per_cell=(9, 9),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
            feature_set = np.concatenate((feature_set, fd), axis=None)
        
        hog_train.append(feature_set)
    
    for i_testing_data in range(len(testData)):
        feature_test_set = []

        for i_view in range(len(testData[i_testing_data])):
            resized_img = cv2.resize(cv2.imread(testData[i_testing_data][i_view][0]), (42, 42))
            fd, _ = hog(resized_img, orientations=9, pixels_per_cell=(9, 9),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
            feature_test_set = np.concatenate((feature_test_set, fd), axis=None)

        hog_test.append(feature_test_set)
    
    return hog_train, hog_test

def svm(hog_train, hog_test):
    y = []
    for i in range(len(trainData)):
        y.append(trainData[i][0][1])

    #train classification model
    print("\nTrain the SVM model")
    clf=LinearSVC(max_iter=80000)    
    clf.fit(hog_train, y)

    y_test = []
    for i in range(len(testData)):
        y_test.append(testData[i][0][1])

    #run on the trained model
    print("\nRunning testing seeds on the trained SVM model...")
    y_pred = clf.predict(hog_test)

    #assign the actual name of seed classes instead of 0 and 1
    true_classes_hog=[]
    for i in y_test:
        if i==1:
           true_classes_hog.append("GoodSeed")
        else:
           true_classes_hog.append("BadSeed")

    predict_classes_hog=[]
    for i in y_pred:
        if i==1:
           predict_classes_hog.append("GoodSeed")
        else:
           predict_classes_hog.append("BadSeed")

    return y_test, y_pred, true_classes_hog, predict_classes_hog

def evaluate_hog(hog_train, hog_test):
    y_test, y_pred, true_classes_hog, predict_classes_hog = svm(hog_train, hog_test)

    print("\nHOG")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names = ['Bad Seeds','Good Seeds']))
    
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=range(2)))

    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG'
    path_csv_hog = os.path.join(path, 'classification_results.csv')
    
    image_paths_test = []
    for i in np.array(testData)[:, 0, 0]:
        image_paths_test.append(i)
    
    false_score_good_hog, false_score_bad_hog, true_score_good_hog, true_score_bad_hog, total_bad_seeds_hog, total_good_seeds_hog = save_results_csv(path, path_csv_hog, true_classes_hog, predict_classes_hog, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds_hog)
    print("No. of Bad seeds detected correctly: ", true_score_bad_hog)
    print("No. of Bad seeds detected wrongly: ", false_score_bad_hog)

    print("\nTotal good testing seeds: ",total_good_seeds_hog)
    print("No. of Good seeds detected correctly: ", true_score_good_hog)
    print("No. of Good seeds detected wrongly: ", false_score_good_hog)

    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG/Good_seeds/' 

    seed_set = []
    seed_type = []

    for i in np.array(image_paths_test):
        seed_set.append(i.split('/')[-3].split('S')[1]) # set of seeds
        seed_type.append(i.split('/')[-4]) # type of seeds

    save_results_corr_image(path_to_results_bad_seeds,path_to_results_good_seeds, len(testData), y_pred, seed_set, seed_type)