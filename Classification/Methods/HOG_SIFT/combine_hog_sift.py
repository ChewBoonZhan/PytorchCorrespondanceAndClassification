
from cgi import test
import csv
import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/") 


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import numpy as np

from save_results_image import save_results_image
from save_results_csv import save_results_csv


def combine_hog_sift(train_hog_features, test_hog_features, train_features, test_features, train_labels, test_labels, image_paths_test):


    train_hog_sift_features = []
    test_hog_sift_features = []

    print("\nCombining HOG and SIFT bag of features of Training datasets...")
    #append HOG and SIFT training bag of features 
    print("Train features length: ",len(train_features))
    for i in range(len(train_features)):
        train_hog_sift_feature = []
        train_hog_sift_feature = np.concatenate((train_hog_sift_feature, train_hog_features[i]), axis=None)
        train_hog_sift_feature = np.concatenate((train_hog_sift_feature, train_features[i]), axis=None)
        train_hog_sift_features.append(train_hog_sift_feature)
        
    train_hog_sift_features_vstack=train_hog_sift_features[0]
    for descriptor in train_hog_sift_features[1:]:
        train_hog_sift_features_vstack=np.vstack((train_hog_sift_features_vstack,descriptor)) #stack descriptors of each image vertically

    stdslr=StandardScaler().fit(train_hog_sift_features_vstack)
    train_hog_sift_features_vstack=stdslr.transform(train_hog_sift_features_vstack)
    print(train_hog_sift_features_vstack.shape)

    print("\nCombining HOG and SIFT bag of features of Testing datasets...")   
    #append HOG and SIFT testing bag of features 
    print("Test features length: ",len(test_features))
    for i in range(len(test_features)):
        test_hog_sift_feature = []
        test_hog_sift_feature = np.concatenate((test_hog_sift_feature, test_hog_features[i]), axis=None)
        test_hog_sift_feature = np.concatenate((test_hog_sift_feature, test_features[i]), axis=None)
        test_hog_sift_features.append(test_hog_sift_feature)
        
    test_hog_sift_features_vstack=test_hog_sift_features[0]
    for descriptor in test_hog_sift_features[1:]:
        test_hog_sift_features_vstack=np.vstack((test_hog_sift_features_vstack,descriptor)) #stack descriptors of each image vertically

    test_hog_sift_features_vstack=stdslr.transform(test_hog_sift_features_vstack)
    print(test_hog_sift_features_vstack.shape)

    #train and run model
    svm_run(train_hog_sift_features_vstack,test_hog_sift_features_vstack,train_labels,test_labels,image_paths_test)
    

def svm_run(train_hog_sift_features_vstack,test_hog_sift_features_vstack,train_labels,test_labels,image_paths_test):

    print("\nTraining the SVM model...")
    clf=LinearSVC(max_iter=80000)
    clf.fit(train_hog_sift_features_vstack,np.array(train_labels))

    print("\nTesting the trained SVM model...")
    pred_hog_sift_labels = clf.predict(test_hog_sift_features_vstack)
    
    true_classes_hog_sift=[]
    for i in test_labels:
        if i==1:
            true_classes_hog_sift.append("GoodSeed")
        else:
            true_classes_hog_sift.append("BadSeed")

    predict_classes_hog_sift=[]
    for i in pred_hog_sift_labels:
        if i==1:
            predict_classes_hog_sift.append("GoodSeed")
        else:
            predict_classes_hog_sift.append("BadSeed")
    
    #evaluate result
    evaluate_combined(test_labels,pred_hog_sift_labels,true_classes_hog_sift,predict_classes_hog_sift,image_paths_test)


def evaluate_combined(test_labels,pred_hog_sift_labels,true_classes_hog_sift,predict_classes_hog_sift,image_paths_test):
    
    print("\nHOG + SIFT")

    print("\nClassification Report")
    print(classification_report(test_labels, pred_hog_sift_labels, target_names = ['Bad Seeds','Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(test_labels, pred_hog_sift_labels, labels=range(2)))

    #save to CSV
    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined'
    path_csv_hog = os.path.join(path, 'classification_results.csv')
    false_score_good, false_score_bad, true_score_good, true_score_bad, total_bad_seeds, total_good_seeds = save_results_csv(path, path_csv_hog, true_classes_hog_sift, predict_classes_hog_sift, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds)
    print("No. of Bad seeds detected correctly: ", true_score_bad)
    print("No. of Bad seeds detected wrongly: ", false_score_bad)

    print("\nTotal good testing seeds: ",total_good_seeds)
    print("No. of Good seeds detected correctly: ", true_score_good)
    print("No. of Good seeds detected wrongly: ", false_score_good)

    #save as images
    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined/Good_seeds/'
    save_results_image(path_to_results_bad_seeds,path_to_results_good_seeds, pred_hog_sift_labels, image_paths_test)

