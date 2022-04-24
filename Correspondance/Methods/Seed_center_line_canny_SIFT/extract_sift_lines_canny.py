import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
import os

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from draw_lines import draw_lines

sift = cv2.SIFT_create()

# Define function to extract sift feature points from a given image
def extract_sift_lines_canny(imagepath, bbobPath):
  #print('Extracting SIFT features points for {}'.format(imagepath))

  image = cv2.imread(imagepath)

  # draw lines over seeds
  image_lines_overlay = image.copy()
  image_lines_overlay = draw_lines(image_lines_overlay, bbobPath)

  #edge detection
  cannyImage = cv2.Canny(image_lines_overlay, 200, 230)

  cannyImage = cv2.cvtColor(cannyImage,cv2.COLOR_GRAY2RGB)

  cannyImage = cannyImage*2

  image_lines_overlay = image_lines_overlay + cannyImage

  image_lines_overlay = image_lines_overlay.astype(np.uint8)

  # convert to greyscale
  img_gray = cv2.cvtColor(image_lines_overlay, cv2.COLOR_BGR2GRAY)

  # detect features from the image
  keypoints, descriptors = sift.detectAndCompute(img_gray, None) #built-in function
  # keypoints = 1
  # descriptors = 1

  # draw the detected key points
  sift_image = cv2.drawKeypoints(img_gray, keypoints, img_gray)
  # sift_image = 1
  
  return image, sift_image, keypoints, descriptors #return 4 values
