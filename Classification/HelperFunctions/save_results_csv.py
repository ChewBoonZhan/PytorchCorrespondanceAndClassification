import csv
import os


def save_results_csv(path,path_to_csv,true_classes, predict_classes, image_path_test):
    if not os.path.exists(path_to_csv):
       os.mkdir(path)

    false_score_good=0 #wrongly detected seeds for good seeds
    false_score_bad=0 #wrongly detected seeds for bad seeds
    true_score_good=0 #correctly detected seeds for good seeds
    true_score_bad=0 #correctly detected seeds for bad seeds
    total_bad_seeds=0 #total number of bad seeds
    total_good_seeds=0 #total number of good seeds

    print("\nSaving classification results as CSV file to directory...")

    with open(path_to_csv, 'w', newline='') as file: 
       writer = csv.writer(file)
       writer.writerow(["img","true classes", "predicted classes", "accuracy"]) # header
      
       # create test data with the good seeds
       for i in range(len(true_classes)):
         if true_classes[i] == predict_classes[i]: #prediction is correct

           #write to file
           writer.writerow([image_path_test[i],true_classes[i], predict_classes[i],"True"]) 
           
           if(true_classes[i]=='BadSeed'):
             true_score_bad+=1
             total_bad_seeds+=1
           elif(true_classes[i]=='GoodSeed'):
             true_score_good+=1
             total_good_seeds+=1

         else: #prediction is wrong

           #write to file
           writer.writerow([image_path_test[i],true_classes[i], predict_classes[i],"False"])
           if(true_classes[i]=='BadSeed'):
             false_score_bad+=1
             total_bad_seeds+=1
           elif(true_classes[i]=='GoodSeed'):
             false_score_good+=1
             total_good_seeds+=1
    
    return false_score_good, false_score_bad, true_score_good, true_score_bad, total_bad_seeds, total_good_seeds


