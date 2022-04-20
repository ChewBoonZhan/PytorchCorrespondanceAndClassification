import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

#importing required libraries
from dataset import *
import numpy as np
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

trainData = loadTrainData()
testData = loadTestData()

def extract_pixel_feature():
    pixel_train = []
    pixel_test = []
    for i_training_data in range(len(trainData)):
        feature_set = []
        for i_view in range(len(trainData[i_training_data])):
            resized_img = cv2.imread(trainData[i_training_data][i_view][0])
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) #get image based on path
            feature_set = np.concatenate((feature_set, img.flatten()), axis=None)
        pixel_train.append(feature_set)

    pixel_train = np.array(pixel_train)

    for i_testing_data in range(len(testData)):
        feature_set = []
        for i_view in range(len(testData[i_testing_data])):
            resized_img = cv2.imread(testData[i_testing_data][i_view][0])
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) #get image based on path
            feature_set = np.concatenate((feature_set, img.flatten()), axis=None)
        pixel_test.append(feature_set)

    pixel_test = np.array(pixel_test)
    return pixel_train, pixel_test

def svm_method(pixel_train, pixel_test):
    #loaded_model = pickle.load(open('svm_model_pixel.sav','rb'))

    y = []
    for i in range(len(trainData)):
        y.append(trainData[i][0][1])

    clf=svm.LinearSVC(max_iter=80000)
    clf.fit(pixel_train, y)

    y_in_test = []
    for i in range(len(testData)):
        y_in_test.append(testData[i][0][1])

    testy = clf.predict(pixel_test)
    # result = loaded_model.score(pixel_test, y_in_test)
    # print(result)
    return y_in_test, testy

def print_result(pixel_train, pixel_test):
    y_in_test, testy = svm_method(pixel_train, pixel_test)
    test_result = pd.concat([
        pd.DataFrame(np.array(y_in_test)), pd.DataFrame(testy)], axis=1, join="inner")
    test_result.columns = ['actual','predict']
    test_result['accuracy'] = np.where(test_result['actual']== test_result['predict'], True, False)
    print(test_result)
    print("\nClassification Report")
    print(classification_report(y_in_test, testy, target_names = ['Bad Seeds','Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_in_test, testy, labels=range(2)))

if __name__ == '__main__':
    # called when runned from command prompt
    pixel_train, pixel_test = extract_pixel_feature()
    print_result(pixel_train, pixel_test)
    print("Done")
    