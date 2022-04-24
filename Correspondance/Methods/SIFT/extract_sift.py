import cv2
import numpy as np
from matplotlib import pyplot as plt

sift = cv2.SIFT_create()

# Define function to extract sift feature points from a given image
def extract_sift(imagepath):
  
  #print('Extracting SIFT features points for {}'.format(imagepath))

  image = cv2.imread(imagepath)

  # convert to greyscale
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # detect features from the image
  keypoints, descriptors = sift.detectAndCompute(img_gray, None) #built-in function

  # draw the detected key points
  sift_image = cv2.drawKeypoints(img_gray, keypoints, img_gray)

  #+ sift_image to return
  return image, sift_image, keypoints, descriptors #return 4 values
