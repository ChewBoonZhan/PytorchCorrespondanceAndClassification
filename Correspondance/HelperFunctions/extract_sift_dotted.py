import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

sift = cv2.xfeatures2d.SIFT_create()

# Define function to extract sift feature points from a given image
def extract_sift_dotted(imagepath, bbobPath):
  print('Extracting SIFT features points for {}'.format(imagepath))

  image = cv2.imread(imagepath)

  radius = 20
  color = (255,0,0)

  # Line thickness of 2 px
  thickness = 20

  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(bbobPath)

  numSeeds = x_min.shape[0]

  for index in range(numSeeds):
    xCenter = int((x_min[index] + x_max[index])/2)
    yCenter = int((y_min[index] + y_max[index])/2)

    image = cv2.circle(image, (xCenter,yCenter), radius, color, thickness)

  # convert to greyscale
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # detect features from the image
  keypoints, descriptors = sift.computeKeypointsAndDescriptors(img_gray) #built-in function

  # draw the detected key points
  sift_image = cv2.drawKeypoints(img_gray, keypoints, img_gray)

  #+ sift_image to return

  return image, sift_image, keypoints, descriptors #return 4 values
