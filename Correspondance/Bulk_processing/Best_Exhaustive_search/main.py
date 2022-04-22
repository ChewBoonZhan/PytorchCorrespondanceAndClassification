import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0, os.getcwd() + "/../../Methods/Best_Exhaustive_search")
sys.path.insert(0, os.getcwd() + "/../../HelperFunctions")

from merge_images_v import merge_images_v

from exhaustive_search import exhaustive_search
from find_correspondance_rotated import find_correspondance_rotated
from extractIndividualSeeds import extract_seeds
from createCorrespondanceAccuracyFile import createCorrespondanceAccuracyFile
from calculateAccuracyCorrespondance import calculateAccuracyCorrespondance

import cv2

import numpy as np

if __name__ == '__main__':
    # called when runned from command prompt

    createCorrespondanceAccuracyFile()
    methodUsed = "Best_Exhaustive_search"
    source_orientation = "front"  
    dest_orientation_list = ["right","left","top","rear"]  
    for i in (n+1 for n in range(12)): #set number 1 to 12
        # Bad seeds
        setNum = i
        seedType = "Bad"    

        for dest_orientation in dest_orientation_list:
            (image1, image2, boundingBoxCollection, transformedBoundingBox, rotationMatrixCollection, paddingImagesCollection, cLabel) = exhaustive_search(
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType+ "_seeds/S" + str(setNum) + "/" + dest_orientation + "_S" + str(setNum) + ".jpg", 
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "_S" + str(setNum) + ".jpg", 
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", 
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", 
                dest_orientation, source_orientation)
            # as can be seen, with exhaustive search (top) the image is much more aligned on top of one another
            # without exhaustive search (bottom) the image looks worse

            # cant run the bottom one, cause it'll call "create_file"
            #image 2 - source
            #image 1 - destination 
            image1, image2 = find_correspondance_rotated(image1, image2, boundingBoxCollection, rotationMatrixCollection, paddingImagesCollection, os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", cLabel, dest_orientation, source_orientation, setNum, seedType, True)
            path_to_results = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seedType  + '_seeds'+ '/S' + str(setNum) + '/Results_' + methodUsed + '/'
            isExist = os.path.exists(path_to_results)

            if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(path_to_results)
                print("The new directory is created!")
            cv2.imwrite(os.path.join(path_to_results, 'correspondence_' + source_orientation + "2" + dest_orientation + ".jpg"), merge_images_v(image2, image1))

            print("Saving images of correspondence for ", seedType, " Set ", str(setNum), " ", source_orientation, " with ", dest_orientation, " to folder")


    for i in (n+1 for n in range(10)): #set number 1 to 12
        # "Good_seeds"
        setNum = i
        seedType = "Good"    

        for dest_orientation in dest_orientation_list:
            (image1, image2, boundingBoxCollection, transformedBoundingBox, rotationMatrixCollection, paddingImagesCollection, cLabel) = exhaustive_search(
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType+ "_seeds/S" + str(setNum) + "/" + dest_orientation + "_S" + str(setNum) + ".jpg",  
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "_S" + str(setNum) + ".jpg", 
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", 
                os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", 
                dest_orientation, source_orientation)
            # as can be seen, with exhaustive search (top) the image is much more aligned on top of one another
            # without exhaustive search (bottom) the image looks worse

            # cant run the bottom one, cause it'll call "create_file"
            image1, image2 = find_correspondance_rotated(image1, image2, boundingBoxCollection, rotationMatrixCollection, paddingImagesCollection, os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", cLabel, dest_orientation, source_orientation, setNum, seedType, True)
            path_to_results = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seedType + '_seeds'+ '/S' + str(setNum) + '/Results_' + methodUsed + '/'
            isExist = os.path.exists(path_to_results)

            if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(path_to_results)
                print("The new directory is created!")

            cv2.imwrite(os.path.join(path_to_results, 'correspondence_' + source_orientation + "2" + dest_orientation + ".jpg"), merge_images_v(image2, image1))

            print("Saving images of correspondence for ", seedType, " Set ", str(setNum), " ", source_orientation, " with ", dest_orientation, " to folder")
    calculateAccuracyCorrespondance()
    
    #crop individual seeds out based on correspondence
    extract_seeds()