import os
import sys
import cv2

from cropSideNoise import cropSideNoise

sys.path.insert(0, os.getcwd() + "/../General_Helper_Function/")

from loadImage import loadImage

if __name__ == '__main__':
    #define a list of view angles
    view = ["top", "right", "left", "front", "rear"]

    #bad seeds
    for i in (n+1 for n in range(12)): #for each set from 1 to 12
        i = str(i)
        print("Cropping Bad Seeds Set ", i)

        #set path to save cropped images for each set
        path_bad = os.getcwd() + '/../Data/ProcessedData/SIFT_try/Bad_seeds/S' + i + '/'   #example: SIFT_try/Bad_seeds/S1/
        isExist = os.path.exists(path_bad)
        #if directory doesnt exist
        if not isExist: 
            # Create a new directory 
            os.makedirs(path_bad)

        for j in range(len(view)):   #for each view
            print(" ",view[j])
            img = loadImage("Bad_seeds", view[j], i) #load original seed image from directory
            cropped_img = cropSideNoise(img, "bad_seeds", view[j], i) #crop 
            name = view[j] + '_S' + i + '.jpg'  #name the cropped image, example: top_S2.jpg
            if cropped_img.any():
                cv2.imwrite(os.path.join(path_bad, name),cropped_img) #save it to the path
            else:
                print("Error ", view[j])


    #good seeds
    for i in (n+1 for n in range(10)): #set number 1 to 10
        i = str(i)
        print("Cropping Good Seeds Set ", i)

        path_good = os.getcwd() + '/../Data/ProcessedData/SIFT_try/Good_seeds/S' + i + '/'
        isExist = os.path.exists(path_good)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_good)
            
        for j in range(len(view)): #view
            print(" ",view[j])
            img = loadImage("Good_seeds", view[j], i)
            cropped_img = cropSideNoise(img, "good_seeds", view[j], i)
            name = view[j] + '_S' + i + '.jpg'  #example: top_S2.jpg
            if cropped_img.any():
                cv2.imwrite(os.path.join(path_good, name),cropped_img) #save it to the path
            else:
                print("Error ", view[j])
        