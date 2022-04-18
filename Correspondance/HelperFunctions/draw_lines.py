import cv2

# Draw connecting lines among seeds

import os
import sys
sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV

def draw_lines(image, bbobPath, thickness = 20, color = (255,0,0)):
  radius = 20

  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(bbobPath)

  numSeeds = x_min.shape[0]

  #draw lines using bounding box coordinates of each seed
  for index in range(numSeeds):
    currentX_min = x_min[index]
    currentY_min = y_min[index]
    currentY_max = y_max[index]
    currentX_max = x_max[index]

    centerX = int((currentX_min + currentX_max)/2)
    centerY = int((currentY_min + currentY_max)/2)

    for index2 in range(numSeeds):
      if(index2 == index):
        q = 1
      else:
        x_minIndex = x_min[index2]
        x_maxIndex = x_max[index2]
        y_minIndex = y_min[index2]
        y_maxIndex = y_max[index2]

        centerXCompare = int((x_minIndex + x_maxIndex)/2)
        centerYCompare = int((y_minIndex + y_maxIndex)/2)
        #draw lines
        image = cv2.line(image, (centerX, centerY), (centerXCompare, centerYCompare), color, thickness)
        
  return image
