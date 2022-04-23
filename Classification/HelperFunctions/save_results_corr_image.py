import os
import sys
import cv2

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV

def save_results_corr_image(path_to_results_bad_seeds, path_to_results_good_seeds, len_testData, predict_labels, seed_set, seed_type):
    isExist_bad = os.path.exists(path_to_results_bad_seeds)
    isExist_good = os.path.exists(path_to_results_good_seeds)

    if not isExist_bad:
        # Create a new directory because it does not exist 
        os.mkdir(path_to_results_bad_seeds)
        print("The new directory is created!")

    if not isExist_good:
        # Create a new directory because it does not exist 
        os.mkdir(path_to_results_good_seeds)
        print("The new directory is created!")

    #set paths to get the seed images (with surrounding cropped) and bbox
    path_to_badseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Bad_seeds/S'
    path_to_goodseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Good_seeds/S'
    path_to_bbox_badseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/BBOX/Bad_seeds/S'
    path_to_bbox_goodseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/BBOX/Good_seeds/S'

    view=["top","right","left","front","rear"]

    good=False
    print("\nSaving classification images to directory...")

    i=0
    while i < len_testData:
    #set paths to img and bbox according to the set index
        if(seed_type[i] == "Good_seeds"): #good seeds 9 and 10
            img_path = path_to_goodseeds + str(seed_set[i]) + '/'
            bbox_path = path_to_bbox_goodseeds + str(seed_set[i]) + '/'
            good = True
        else: #bad seeds 10, 11, and 12
            img_path = path_to_badseeds + str(seed_set[i]) + '/'
            bbox_path = path_to_bbox_badseeds + str(seed_set[i]) + '/'
            good = False

        for j in range(len(view)):
            print("Saving Classification Set", str(seed_set[i]), str(seed_type[i]) ,view[j])
            img_path_view = img_path + view[j] + '_S' + str(seed_set[i]) + '.jpg'
            bbox_path_view = bbox_path +  view[j] + '/'
            x_min, y_min, x_max, y_max = readBoundingBoxCSV(bbox_path_view, True)
            numberOfSeeds = x_max.shape[0]
            img=cv2.imread(img_path_view)
            
            for index in range(numberOfSeeds): #for each seed in the seed view image
                #get its predicted label      
                pred_label = predict_labels[i+index]
                
                #retrieve its bounding box coordinates
                x_minIndex = x_min[index]
                y_minIndex = y_min[index]
                x_maxIndex = x_max[index]
                y_maxIndex = y_max[index]
                start_point = (x_minIndex, y_minIndex)
                end_point = (x_maxIndex, y_maxIndex)

                xCenter = int(abs(x_minIndex + x_maxIndex)/2)-40
                yCenter = int(abs(y_minIndex + y_maxIndex)/2)+40

                if(pred_label==1): #predicted GoodSeed - green colour
                    color=(0,255,0)
                    text='good'
                else: #predicted BadSeed - red colour
                    color=(0,0,255)
                    text='bad'

                #draw bounding box around the seed based on its coordinates + colour according to the predicted label
                img = cv2.rectangle(img, start_point, end_point, color, 10)
                #add label (1,2,3...) for testing only
                img = cv2.putText(img, text, (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 10, cv2.LINE_AA)
            
            #after drawing bbox for each seed in the view image, save it to the directory
            name= view[j] + '_S' + str(seed_set[i]) + '.jpg'
            if(good):
                cv2.imwrite(os.path.join(path_to_results_good_seeds, name),img)
            else:
                cv2.imwrite(os.path.join(path_to_results_bad_seeds, name),img)
        
        i = i + numberOfSeeds

    print("\nSaved.")

