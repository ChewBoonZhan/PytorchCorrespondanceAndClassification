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
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

feature = []
hog_im = []
feature_test  = []
hog_test_im = []
trainData = loadTrainData()
testData = loadTestData()

def hog_method():
    global feature, hog_im, feature_test, hog_test_im

    for i_training_data in range(len(trainData)):
        feature_set  = []
        hog_set = []
    
        for i_view in range(len(trainData[i_training_data])):
            resized_img = cv2.resize(cv2.imread(trainData[i_training_data][i_view][0]), (40, 40))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(9, 9),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
            feature_set = np.concatenate((feature_set, fd), axis=None)
            hog_set = np.concatenate((hog_set, hog_image), axis=None)
        
        feature.append(feature_set)
        hog_im.append(hog_set)

    feature = np.array(feature)
    hog_im = np.array(hog_im)
    
    for i_testing_data in range(len(testData)):
        feature_test_set = []
        hog_test_set = []

        for i_view in range(len(testData[i_testing_data])):
            resized_img = cv2.resize(cv2.imread(testData[i_testing_data][i_view][0]), (40, 40))
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(9, 9),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
            feature_test_set = np.concatenate((feature_test_set, fd), axis=None)
            hog_test_set = np.concatenate((hog_test_set, hog_image), axis=None)

        feature_test.append(feature_test_set)
        hog_test_im.append(hog_test_set)
    
    feature_test = np.array(feature_test)
    hog_test_im = np.array(hog_test_im)

def svm_method():
    y = []
    for i in range(len(trainData)):
        y.append(trainData[i][0][1])

    # clf = svm.NuSVC(kernel='sigmoid',gamma='auto')
    clf = svm.LinearSVC()
    clf.fit(feature, y)

    y_in_test = []
    for i in range(len(testData)):
        y_in_test.append(testData[i][0][1])

    testy = clf.predict(feature_test)
    return y_in_test, testy

def print_result():
    y_in_test, testy = svm_method()
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
    hog_method()
    print_result()
    print("Done")
    