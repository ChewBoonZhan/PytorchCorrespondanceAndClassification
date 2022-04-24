
import os
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from get_homography import get_homography
from extract_sift import extract_sift

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV
from form_corresponding_bounding_boxes import form_corresponding_bounding_boxes

if __name__ == '__main__':

    #extract sift features of source and destination images
    image1, sift_image1, keypoints1, descriptors1 = extract_sift(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Bad_seeds/S3/right_S3.jpg")
    image2, sift_image2, keypoints2, descriptors2 = extract_sift(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Bad_seeds/S3/top_S3.jpg")

    sift_src_dict = {'keypoints_src': keypoints1, 'descriptors_src': descriptors1}
    sift_dst_dict = {'keypoints_dst': keypoints2, 'descriptors_dst': descriptors2}

    #compute homography matrix based on the sift features extracted
    homoMatrix = get_homography(sift_src_dict, sift_dst_dict)

    #retrieve bounding box (bbox) info of source image
    (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/Bad_seeds/S3/right/")

    boundingBox1 = {
        "x_min":x_min,
        "y_min":y_min,
        "x_max":x_max,
        "y_max":y_max
    }

    #apply homography matrix on the bbox of the source image to get the transformed bbox drawn on the destination image
    imageOut1, imageOut2 = form_corresponding_bounding_boxes(image1, image2, homoMatrix, boundingBox1)
    #can save

    # Show different images as result.
    f, axarr = plt.subplots(1, 2)
    axarr[0].axis('off')
    axarr[0].set_title("Source Image")
    axarr[0].imshow(imageOut1)

    axarr[1].imshow(imageOut2)
    axarr[1].axis('off')
    axarr[1].set_title("Destination Image")

    plt.show()


