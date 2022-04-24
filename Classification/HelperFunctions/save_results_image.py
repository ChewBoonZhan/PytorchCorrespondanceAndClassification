import os
import sys
import cv2

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV


def save_results_image(path_to_results_bad_seeds, path_to_results_good_seeds, predict_labels, image_paths):

    isExist_bad = os.path.exists(path_to_results_bad_seeds)
    isExist_good = os.path.exists(path_to_results_good_seeds)
    
    if not isExist_bad:
        # Create a new directory because it does not exist 
        os.makedirs(path_to_results_bad_seeds)
        print("The new directory is created!")

    if not isExist_good:
        # Create a new directory because it does not exist 
        os.makedirs(path_to_results_good_seeds)
        print("The new directory is created!")

    #set paths to get the seed images (with surrounding cropped) and bbox
    path_to_badseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Bad_seeds/S'
    path_to_goodseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Good_seeds/S'
    path_to_bbox_badseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/BBOX/Bad_seeds/S'
    path_to_bbox_goodseeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/BBOX/Good_seeds/S'

    view=["top","right","left","front","rear"]


    numberOfSeeds_prev=0
    good=False
    print("\nSaving classification images to directory...")

    for i in (n+10 for n in range(5)): #for each testing seed set

       #set paths to img and bbox according to the set index
       if(i>12): #good seeds set 9 and 10
          i=i-4
          img_path = path_to_goodseeds + str(i) + '/'
          bbox_path = path_to_bbox_goodseeds + str(i) + '/'
          good = True
          seed_type="Good_seeds"
       else: #bad seeds set 10,11,12
           img_path = path_to_badseeds + str(i) + '/'
           bbox_path = path_to_bbox_badseeds + str(i) + '/'
           good = False
           seed_type="Bad_seeds"

       for j in range(len(view)): #for each view of the seed set 

          #set path to the view image
          img_path_view = img_path + view[j] + '_S' + str(i) + '.jpg'
          img=cv2.imread(img_path_view)
          print("Saving Classification Set ",str(i), seed_type, view[j] )

          #set path to the bbox of the view image
          bbox_path_view = bbox_path + view[j] + '/'
          x_min, y_min, x_max, y_max = readBoundingBoxCSV(bbox_path_view)

          numberOfSeeds = x_max.shape[0]
          
          for index in range(numberOfSeeds): #for each seed in the seed view image
              
              #retrieve its bounding box coordinates
              x_minIndex = x_min[index]
              y_minIndex = y_min[index]
              y_maxIndex = y_max[index]
              x_maxIndex = x_max[index]
              
              start_point = (x_minIndex, y_minIndex)
              end_point = (x_maxIndex, y_maxIndex)
              
              xCenter = int(abs(x_minIndex + x_maxIndex)/2)-40
              yCenter = int(abs(y_minIndex + y_maxIndex)/2)+40

              #get its predicted label
              pred_label = predict_labels[(5*numberOfSeeds_prev)+j+(index*5)]
              
              #check if we're retrieving from the correct image 
              #test_img = image_paths[(5*numberOfSeeds_prev)+j+(index*5)]
              #print(test_img)
              #print(pred_label,"Set ",str(i)," ",view[j], " Seed ",index+1)

              if(pred_label==1): #predicted GoodSeed - green colour
                color=(0,255,0)
                text='good'
              else: #predicted BadSeed - red colour
                color=(0,0,255)
                text='bad'

              #draw bounding box around the seed based on its coordinates + colour according to the predicted label
              img = cv2.rectangle(img, start_point, end_point, color, 10)
              #add label good/bad
              img = cv2.putText(img, text, (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 10, cv2.LINE_AA)
      
          #after drawing bbox for each seed in the view image, save it to the directory
          name= view[j] + '_S' + str(i) + '.jpg'
          if(good):
            cv2.imwrite(os.path.join(path_to_results_good_seeds, name),img)
          else:
            cv2.imwrite(os.path.join(path_to_results_bad_seeds, name),img)

       numberOfSeeds_prev = numberOfSeeds_prev + numberOfSeeds

    print("\nSaved.")

