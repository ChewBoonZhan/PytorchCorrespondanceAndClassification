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

  print('Finding correspondence for ' , seed_type, ' Set ', set)

  #set image path to the CROPPED images directory
  #assuming the cropped images are saved to SIFT_try folder
  src_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/top_S' + set +'.jpg'
  right_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/right_S' + set +'.jpg'
  left_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/left_S' + set +'.jpg'
  front_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/front_S' + set +'.jpg'
  rear_img_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/rear_S' + set +'.jpg'

  #set paths to bounding box csv
  #example: SIFT_try/BBOX/Bad_seeds/S2/top/
  src_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/top/'
  right_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/right/'
  left_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/left/'
  front_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/front/'
  rear_bbox_path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/' + seed_type + '/S' + set + '/rear/'

  #if original method
  image_src, sift_image_src, keypoints_src, descriptors_src = extract_sift(src_img_path) #top
  image_right, sift_image_right, keypoints_right, descriptors_right = extract_sift(right_img_path) 
  image_left, sift_image_left, keypoints_left, descriptors_left = extract_sift(left_img_path) 
  image_front, sift_image_front, keypoints_front, descriptors_front = extract_sift(front_img_path) 
  image_rear, sift_image_rear, keypoints_rear, descriptors_rear = extract_sift(rear_img_path) 

  #save in a dict to return -> can be used for classfication training later
  sift_src_dict = {'keypoints_src': keypoints_src, 'descriptors_src': descriptors_src}
  sift_right_dict = {'keypoints_dst': keypoints_right, 'descriptors_dst': descriptors_right}
  sift_left_dict = {'keypoints_dst': keypoints_left, 'descriptors_dst': descriptors_left}
  sift_front_dict = {'keypoints_dst': keypoints_front, 'descriptors_dst': descriptors_front}
  sift_rear_dict = {'keypoints_dst': keypoints_rear, 'descriptors_dst': descriptors_rear}

  #get homography matrix based on SIFT
  #need to manipulate get_homography input arguments to take in dicts to retrieve
  homoMatrix_Top2Right = get_homography(sift_src_dict,sift_right_dict)
  homoMatrix_Top2Left = get_homography(sift_src_dict,sift_left_dict)
  homoMatrix_Top2Front = get_homography(sift_src_dict,sift_front_dict)
  homoMatrix_Top2Rear = get_homography(sift_src_dict,sift_rear_dict)


  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(src_bbox_path)

  #(x_min, y_min, x_max, y_max) = readBoundingBoxCSV("SIFT_try/BBOX/Bad_seeds/S2/top/")

  boundingBox_src = {
    "x_min":x_min,
    "y_min":y_min,
    "x_max":x_max,
    "y_max":y_max
  }


  #form corresponding bounding boxes for each Top x View pair
  imageOut_TopRight, imageOut_Right = form_corresponding_bounding_boxes(image_src, image_right, homoMatrix_Top2Right, boundingBox_src)
  imageOut_TopLeft, imageOut_Left = form_corresponding_bounding_boxes(image_src, image_left, homoMatrix_Top2Left, boundingBox_src)
  imageOut_TopFront, imageOut_Front = form_corresponding_bounding_boxes(image_src, image_front, homoMatrix_Top2Front, boundingBox_src)
  imageOut_TopRear, imageOut_Rear = form_corresponding_bounding_boxes(image_src, image_rear, homoMatrix_Top2Rear, boundingBox_src)


  #save the output images to a "Results" folder, varied according to method
  #set path. Example: 'SIFT_try/Bad_seeds/S2/Results_lines_Canny/xyz.jpg'
  path_to_results = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/'+ seed_type + '/S' + set + '/Results_' + methodUsed + "/"

    # Check whether the specified path exists or not
  isExist = os.path.exists(path_to_results)

  if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(path_to_results)
    print("The new directory is created!")

  #merge and save images to show Results 
  print("Saving images of correspondence for ", seed_type, " Set ", set, " to folder")
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Top2Right.jpg'), merge_images_v(imageOut_TopRight, imageOut_Right))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Top2Left.jpg'), merge_images_v(imageOut_TopLeft, imageOut_Left))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Top2Front.jpg'), merge_images_v(imageOut_TopFront, imageOut_Front))
  cv2.imwrite(os.path.join(path_to_results, 'correspondence_Top2Rear.jpg'), merge_images_v(imageOut_TopRear, imageOut_Rear))


