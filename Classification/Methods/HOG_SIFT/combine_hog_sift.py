
import csv
import os

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from save_results_image import save_results_image

def combine_hog_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test, y_pred,predict_classes_hog):

    print("Combining HOG and SIFT....")

    false_score_good=0
    false_score_bad=0
    true_score_good=0
    true_score_bad=0
    total_bad_seeds=0
    total_good_seeds=0
    predict_class = 0

    path = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined'
    path_csv = os.path.join(path, 'classification_results.csv')

    if not os.path.exists(path_csv):
       os.makedirs(path)

    combined_pred = []

    with open(path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img","true classes", "predicted classes", "accuracy"]) # header

        #loop through each testing seed
        for i in range(len(test_labels)):
            
            if test_labels[i] != pred_labels[i]: #if predict from SIFT is incorrect

               predict_class = y_pred[i] #use predict from hog
               combined_pred.append(predict_class) #update to the new predict labels

               if test_labels[i] == predict_class: #if predict from hog is correct

                  writer.writerow([image_paths_test[i],true_classes_sift[i], predict_classes_hog[i] ,"True"]) # get the filename in the good seed folder, and write each row with filename and 1
          
                  if(true_classes_sift[i]=='BadSeed'):
                      true_score_bad+=1
                      total_bad_seeds+=1
                  elif(true_classes_sift[i]=='GoodSeed'):
                      true_score_good+=1
                      total_good_seeds+=1

               else: #predict from hog is incorrect

                  writer.writerow([image_paths_test[i],true_classes_sift[i], predict_classes_hog[i],"False"])
                  
                  if(true_classes_sift[i]=='BadSeed'):
                      false_score_bad+=1
                      total_bad_seeds+=1
                  elif(true_classes_sift[i]=='GoodSeed'):
                      false_score_good+=1
                      total_good_seeds+=1

            else: #predict from SIFT is correct where test_labels[i] == pred_labels[i]

               writer.writerow([image_paths_test[i],true_classes_sift[i], predict_classes_sift[i] ,"True"])
               combined_pred.append(pred_labels[i]) #update to the new predict labels

               if(true_classes_sift[i]=='BadSeed'):
                   true_score_bad+=1
                   total_bad_seeds+=1
               elif(true_classes_sift[i]=='GoodSeed'):
                   true_score_good+=1
                   total_good_seeds+=1
    
    #evaluate classification accuracy
    evaluate_combined(test_labels, combined_pred, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds)
    print("No. of Bad seeds detected correctly: ", true_score_bad)
    print("No. of Bad seeds detected wrongly: ", false_score_bad)
    
    print("\nTotal good testing seeds: ",total_good_seeds)
    print("No. of Good seeds detected correctly: ", true_score_good)
    print("No. of Good seeds detected wrongly: ", false_score_good)


def evaluate_combined(test_labels, combined_pred, image_paths_test):

    print("\nHOG + SIFT")

    print("\nClassification Report")
    print(classification_report(test_labels, combined_pred, target_names = ['Bad Seeds','Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(test_labels, combined_pred, labels=range(2)))

    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_Combined/Good_seeds/'

    save_results_image(path_to_results_bad_seeds,path_to_results_good_seeds, combined_pred, image_paths_test)
