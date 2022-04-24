import cv2
import os
import sys
from matplotlib.pyplot import figure

sys.path.insert(0, os.getcwd() + "/../../Methods/SIFT")
sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from extract_sift import extract_sift
from get_homography import get_homography
from form_corresponding_bounding_boxes import form_corresponding_bounding_boxes
from merge_images_v import merge_images_v

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV

def find_correspondence( seed_type, set, methodUsed): #Bad seeds Good seeds

  #source: front
  #destination: top, right, left, rear

  print('\nFinding correspondence for ' , seed_type, ' Set ', set)

  #set image path to the CROPPED images (for a set of 5 views) 
  #example: SIFT_try/Bad_seeds/S2/front_S2.jpg
  src_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/front_S' + set +'.jpg'
  top_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/top_S' + set +'.jpg'  
  right_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/right_S' + set +'.jpg'
  left_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/left_S' + set +'.jpg'
  rear_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/rear_S' + set +'.jpg'

  #set paths to bounding box csv of the source
  #example: SIFT_try/BBOX/Bad_seeds/S2/front/
  src_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/front/'

  #extract sift features for all 5 images
  image_src, sift_image_src, keypoints_src, descriptors_src = extract_sift(src_img_path) #top
  image_right, sift_image_right, keypoints_right, descriptors_right = extract_sift(right_img_path) 
  image_left, sift_image_left, keypoints_left, descriptors_left = extract_sift(left_img_path) 
  image_top, sift_image_top, keypoints_top, descriptors_top = extract_sift(top_img_path) 
  image_rear, sift_image_rear, keypoints_rear, descriptors_rear = extract_sift(rear_img_path) 

  #save in a dict to return 
  sift_src_dict = {'keypoints_src': keypoints_src, 'descriptors_src': descriptors_src}
  sift_right_dict = {'keypoints_dst': keypoints_right, 'descriptors_dst': descriptors_right}
  sift_left_dict = {'keypoints_dst': keypoints_left, 'descriptors_dst': descriptors_left}
  sift_top_dict = {'keypoints_dst': keypoints_top, 'descriptors_dst': descriptors_top}
  sift_rear_dict = {'keypoints_dst': keypoints_rear, 'descriptors_dst': descriptors_rear}

  #get homography matrix based on SIFT
  homoMatrix_Front2Right = get_homography(sift_src_dict,sift_right_dict)
  homoMatrix_Front2Left = get_homography(sift_src_dict,sift_left_dict)
  homoMatrix_Front2Top = get_homography(sift_src_dict,sift_top_dict)
  homoMatrix_Front2Rear = get_homography(sift_src_dict,sift_rear_dict)

  #retrieve bounding box csv of the cropped source image (src)
  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(src_bbox_path)

  boundingBox_src = {
    "x_min":x_min,
    "y_min":y_min,
    "x_max":x_max,
    "y_max":y_max
  }


  #form corresponding bounding boxes for each Front x View pair
  imageOut_FrontRight, imageOut_Right = form_corresponding_bounding_boxes(image_src, image_right, homoMatrix_Front2Right, boundingBox_src)
  imageOut_FrontLeft, imageOut_Left = form_corresponding_bounding_boxes(image_src, image_left, homoMatrix_Front2Left, boundingBox_src)
  imageOut_FrontTop, imageOut_Top = form_corresponding_bounding_boxes(image_src, image_top, homoMatrix_Front2Top, boundingBox_src)
  imageOut_FrontRear, imageOut_Rear = form_corresponding_bounding_boxes(image_src, image_rear, homoMatrix_Front2Rear, boundingBox_src)


  #save the output images to a "Results" folder, varied according to method
  #example: 'SIFT_try/Bad_seeds/S2/Results_SIFT/xyz.jpg'
  path_to_results = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/Results_' + methodUsed + "/"

    # Check whether the specified path exists or not
  isExist = os.path.exists(path_to_results)

  if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(path_to_results)
    print("The new directory is created!")

  #merge and save images to show Results 
  print("Saving images of correspondence for ", seed_type, " Set ", set, " to folder")
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Front2Right.jpg'), merge_images_v(imageOut_FrontRight, imageOut_Right))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Front2Left.jpg'), merge_images_v(imageOut_FrontLeft, imageOut_Left))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Front2Top.jpg'), merge_images_v(imageOut_FrontTop, imageOut_Top))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Front2Rear.jpg'), merge_images_v(imageOut_FrontRear, imageOut_Rear))

