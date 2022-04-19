#importing required libraries
import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/") #not working on my side so i use the way below

from save_results_image import save_results_image
from save_results_csv import save_results_csv
from dataset import loadTestData, loadTrainData

import numpy 
import cv2
from skimage.transform import resize
from skimage.feature import hog
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#load training and testing datasets
trainData = loadTrainData()
testData = loadTestData()

def hog_extract():
    
   #training features and hog images
   feature = []
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

   feature = numpy.array(feature)
   hog_im = numpy.array(hog_im)
   print(feature.shape)
   feature = feature.reshape(feature.shape[0],feature.shape[2])

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
  
   feature_test = numpy.array(feature_test)
   hog_test_im = numpy.array(hog_test_im)
   print(feature_test.shape)
   feature_test= feature_test.reshape(feature_test.shape[0],feature_test.shape[2])
   
   #train and test on SVM model
   return svm(feature,feature_test)


def svm(feature,feature_test):

   #train linear SVC model using the training seeds' features and labels
   y_train = []
   for i in range(len(trainData)):
       for j in range(len(trainData[i])):
           y_train.append(trainData[i][j][1])

   #retrieve correct seed labels from the testing seeds
   y_test = []
   image_path_test_hog = []
   for i in range(len(testData)):
       for j in range(len(trainData[i])):
           y_test.append(testData[i][j][1])
           image_path_test_hog.append(testData[i][j][0])
   
   #train model
   print("\nTraining model...")
   clf=LinearSVC(max_iter=80000)
   clf.fit(feature, y_train)
   
   #run test on the trained model
   print("\nTesting on the trained model....")
   y_pred = clf.predict(feature_test)

   #add names to labels
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

   return y_test, y_pred, true_classes_hog, predict_classes_hog, image_path_test_hog



def evaluate_hog(y_test, y_pred, true_classes_hog, predict_classes_hog, image_path_test_hog):
    
    print("\nHOG")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names = ['Bad Seeds','Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred, labels=range(2)))

    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG'
    path_csv_hog = os.path.join(path, 'classification_results.csv')

    false_score_good_hog, false_score_bad_hog, true_score_good_hog, true_score_bad_hog, total_bad_seeds_hog, total_good_seeds_hog = save_results_csv(path, path_csv_hog, true_classes_hog, predict_classes_hog, image_path_test_hog)

    print("\nTotal bad testing seeds: ", total_bad_seeds_hog)
    print("No. of Bad seeds detected correctly: ", true_score_bad_hog)
    print("No. of Bad seeds detected wrongly: ", false_score_bad_hog)

    print("\nTotal good testing seeds: ",total_good_seeds_hog)
    print("No. of Good seeds detected correctly: ", true_score_good_hog)
    print("No. of Good seeds detected wrongly: ", false_score_good_hog)

    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG/Good_seeds/'

    save_results_image(path_to_results_bad_seeds,path_to_results_good_seeds, y_pred, image_path_test_hog)


if __name__ == '__main__':
    # called when runned from command prompt
    hog_extract()

