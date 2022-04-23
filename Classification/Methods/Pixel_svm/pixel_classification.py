import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

#importing required libraries
from dataset import *
import numpy as np
import cv2
import numpy as np
from sklearn.svm import LinearSVC   
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from save_results_csv import save_results_csv
from save_results_corr_image import save_results_corr_image

trainData = loadTrainData()
testData = loadTestData()

def extract_pixel_feature():
    pixel_train = []
    pixel_test = []
    for i_training_data in range(len(trainData)):
        feature_set = []
        for i_view in range(len(trainData[i_training_data])):
            resized_img = cv2.resize(cv2.imread(trainData[i_training_data][i_view][0]), (42, 42))
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature_set = np.concatenate((feature_set, img.flatten()), axis=None)
        pixel_train.append(feature_set)

    pixel_train = np.array(pixel_train)

    for i_testing_data in range(len(testData)):
        feature_set = []
        for i_view in range(len(testData[i_testing_data])):
            resized_img = cv2.resize(cv2.imread(testData[i_testing_data][i_view][0]), (42, 42))
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature_set = np.concatenate((feature_set, img.flatten()), axis=None)
        pixel_test.append(feature_set)

    pixel_test = np.array(pixel_test)
    return pixel_train, pixel_test

def svm(pixel_train, pixel_test):
    y = []
    for i in range(len(trainData)):
        y.append(trainData[i][0][1])
    
    #train classification model
    print("\nTrain the SVM model")
    clf=LinearSVC(max_iter=80000)    
    clf.fit(pixel_train, y)

    y_test = []
    for i in range(len(testData)):
        y_test.append(testData[i][0][1])

    #run on the trained model
    print("\nRunning testing seeds on the trained SVM model...")
    y_pred = clf.predict(pixel_test)

    #assign the actual name of seed classes instead of 0 and 1
    true_classes_pixel=[]
    for i in y_test:
        if i==1:
           true_classes_pixel.append("GoodSeed")
        else:
           true_classes_pixel.append("BadSeed")

    predict_classes_pixel=[]
    for i in y_pred:
        if i==1:
           predict_classes_pixel.append("GoodSeed")
        else:
           predict_classes_pixel.append("BadSeed")

    return y_test, y_pred, true_classes_pixel, predict_classes_pixel

def evaluate_pixel(pixel_train, pixel_test):
    y_test, y_pred, true_classes_pixel, predict_classes_pixel = svm(pixel_train, pixel_test)

    print("\nPixel")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names = ['Bad Seeds','Good Seeds']))
    
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=range(2)))

    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Pixel'
    path_csv_pixel = os.path.join(path, 'classification_results.csv')
    
    image_paths_test = []
    for i in np.array(testData)[:, 0, 0]:
        image_paths_test.append(i)

    false_score_good_pixel, false_score_bad_pixel, true_score_good_pixel, true_score_bad_pixel, total_bad_seeds_pixel, total_good_seeds_pixel = save_results_csv(path, path_csv_pixel, true_classes_pixel, predict_classes_pixel, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds_pixel)
    print("No. of Bad seeds detected correctly: ", true_score_bad_pixel)
    print("No. of Bad seeds detected wrongly: ", false_score_bad_pixel)

    print("\nTotal good testing seeds: ",total_good_seeds_pixel)
    print("No. of Good seeds detected correctly: ", true_score_good_pixel)
    print("No. of Good seeds detected wrongly: ", false_score_good_pixel)

    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Pixel/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Pixel/Good_seeds/' 

    seed_set = []
    seed_type = []

    for i in np.array(image_paths_test):
        seed_set.append(i.split('/')[-3].split('S')[1]) # set of seeds
        seed_type.append(i.split('/')[-4]) # type of seeds

    save_results_corr_image(path_to_results_bad_seeds,path_to_results_good_seeds, len(testData), y_pred, seed_set, seed_type)
    