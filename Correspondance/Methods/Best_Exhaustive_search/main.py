import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0, os.getcwd() + "/../Best_Exhaustive_search")

from exhaustive_search import exhaustive_search
from find_correspondance_rotated import find_correspondance_rotated


import cv2

import numpy as np



if __name__ == '__main__':
    # called when runned from command prompt

    # change variable here to change the output
    setNum = 7           # (1-10) - good seeds, (1-12) - bad seeds
    seedType = "Good"    # Good/Bad

    # right, top, left, rear, front
    source_orientation = "top"  
    dest_orientation = "rear"

    (image1, image2, boundingBoxCollection, transformedBoundingBox, rotationMatrixCollection, paddingImagesCollection, cLabel) = exhaustive_search(
        os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType+ "_seeds/S" + str(setNum) + "/" + source_orientation + "_S" + str(setNum) + ".jpg", os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "_S" + str(setNum) + ".jpg", 
        os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", 
        source_orientation, dest_orientation)
    # as can be seen, with exhaustive search (top) the image is much more aligned on top of one another
    # without exhaustive search (bottom) the image looks worse

    # cant run the bottom one, cause it'll call "create_file"
    image1, image2 = find_correspondance_rotated(image1, image2, boundingBoxCollection, rotationMatrixCollection, paddingImagesCollection, os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + source_orientation + "/", os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "_seeds/S" + str(setNum) + "/" + dest_orientation + "/", cLabel, "", "", "", "", False)

    # Show different images as result.
    f, axarr = plt.subplots(1, 2)
    axarr[0].axis('off')
    axarr[0].set_title("Source Image")
    axarr[0].imshow(image1)

    axarr[1].imshow(image2)
    axarr[1].axis('off')
    axarr[1].set_title("Destination Image")

    plt.show()