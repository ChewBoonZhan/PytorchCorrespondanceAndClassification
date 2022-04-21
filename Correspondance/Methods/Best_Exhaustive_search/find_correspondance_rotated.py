### Find the correspondance of the rotated, translated, scaled images

# the 2 images here are just the cropped images, not rotated, translated, or scaled.

# the bbox are the bbox of cropped images only

# transformation matrices and padding info are obtained from the above computation (exhaustive search function)

import sys
import os
import math
import csv

sys.path.insert(0, os.getcwd())

from createFile import createFile
from writeFile import writeFile

import numpy as np
import cv2

def find_correspondance_rotated(image, image2, boundingBoxCollection, rotationMatrixCollection, paddingImagesCollection, bbobPath1, bbobPath2, cLabelCollection, source_orientation, dest_orientation, setNum, seedType, writeToFile= False):
  correctlyCountedSeeds = 0
  color = (255, 0, 0)
  color2 = (250,218,94)
  thickness = 10

  # seed center from image 2
  boundingBox2 = boundingBoxCollection[1]
  x_minIndexT2 = boundingBox2[0]
  y_minIndexT2 = boundingBox2[1]
  x_maxIndexT2 = boundingBox2[2]
  y_maxIndexT2 = boundingBox2[3]

  numOfSeeds = len(x_minIndexT2)
  seedCenterCollection2 = []
  xCenterCollection2 = []
  yCenterCollection2 = []
  for index in range(numOfSeeds):
    xCenterCollection2.append(((x_minIndexT2[index] + x_maxIndexT2[index])/2))
    yCenterCollection2.append(((y_minIndexT2[index] + y_maxIndexT2[index])/2))

  # center for all the seeds in ori image2
  seedCenterCollection2.append(xCenterCollection2)
  seedCenterCollection2.append(yCenterCollection2)

  # bounding box from ori image 1
  boundingBox1 = boundingBoxCollection[0]
  x_minIndexT = boundingBox1[0]
  y_minIndexT = boundingBox1[1]
  x_maxIndexT = boundingBox1[2]
  y_maxIndexT = boundingBox1[3]
  
  createFile(bbobPath1)
  createFile(bbobPath2)

  for index in range(numOfSeeds):
    # The seed center from image two
    xCenterIndex2 = xCenterCollection2[index]
    yCenterIndex2 = yCenterCollection2[index]
    
    # homo coord for the seed center of ori image 2
    homoSeedCenter = (np.array([xCenterIndex2, yCenterIndex2, 1])).T

    # rotations and transformations performed previously on image 2
    rotationMatrix1 = rotationMatrixCollection[1]
    
    # perform all the rotations on the seed of ori image 2
    rotatedHomoSeed = homoSeedCenter.copy()
    for rotationMatrixIndex in rotationMatrix1:
      rotatedHomoSeed = rotationMatrixIndex @ rotatedHomoSeed

    # Euclideam coordinates with the padding added
    seedX2 = rotatedHomoSeed[0]/rotatedHomoSeed[2] + paddingImagesCollection[1][0]
    seedY2 = rotatedHomoSeed[1]/rotatedHomoSeed[2] + paddingImagesCollection[1][1]


    seedCorrespondanceDetected = False
    for index2 in range(numOfSeeds):   # we check each seed in the ori image 1 to find the seed we are looking for (from ori image 2)
      x_minIndexTIndex = x_minIndexT[index2] 
      y_minIndexTIndex = y_minIndexT[index2]
      x_maxIndexTIndex = x_maxIndexT[index2]
      y_maxIndexTIndex = y_maxIndexT[index2]
     
      # homo coord for the min and max coord of bounding box of ori image 1
      homoPos1 = (np.array([x_minIndexTIndex, y_minIndexTIndex, 1])).T
      homoPos2 = (np.array([x_maxIndexTIndex, y_maxIndexTIndex, 1])).T

      rotatedHomoPos1 = homoPos1.copy()
      rotatedHomoPos2 = homoPos2.copy()

      # perfrom the rotations, scaling on the ori bounding boxes based on what was perfomed previously on image1
      rotationMatrix2 = rotationMatrixCollection[0]
      for rotationMatrixIndex in rotationMatrix2:
        rotatedHomoPos1 = rotationMatrixIndex @ rotatedHomoPos1 
        rotatedHomoPos2 = rotationMatrixIndex @ rotatedHomoPos2 

      # Euclidean for the min, max coord of bounding boxes of the seeds in the ori image after rotation and scaling, then added with padding
      euclidianPos1 = [rotatedHomoPos1[0]/rotatedHomoPos1[2] + paddingImagesCollection[0][0], rotatedHomoPos1[1]/rotatedHomoPos1[2] + paddingImagesCollection[0][1]]
      euclidianPos2 = [rotatedHomoPos2[0]/rotatedHomoPos2[2] + paddingImagesCollection[0][0], rotatedHomoPos2[1]/rotatedHomoPos2[2] + paddingImagesCollection[0][1]]

      # Check if the center of the seed (from ori image 2) that we are trying to find in ori image 1 within the range of the seed bounding box
      if(seedX2 >= min([euclidianPos1[0], euclidianPos2[0]]) and seedX2 <= max([euclidianPos1[0], euclidianPos2[0]]) and seedY2 >= min([euclidianPos1[1], euclidianPos2[1]]) and seedY2 <= max([euclidianPos1[1], euclidianPos2[1]])):
        
        seedCorrespondanceDetected = True
        # it is in the bounding box
        image = cv2.rectangle(image, (x_minIndexT[index2], y_minIndexT[index2]), (x_maxIndexT[index2], y_maxIndexT[index2]), color, thickness)
        image = cv2.putText(image, str(index), (int((x_minIndexT[index2] + x_maxIndexT[index2])/2) - 40, int((y_minIndexT[index2] + y_maxIndexT[index2])/2)+ 40) , cv2.FONT_HERSHEY_SIMPLEX, 3, color2, 10, cv2.LINE_AA)

        # write coordinate for right, left, front, rear
        writeFile(x_minIndexT[index2], y_minIndexT[index2], x_maxIndexT[index2], y_maxIndexT[index2], bbobPath1, numOfSeeds)

        # check if correspondance is detected correctly
        if(cLabelCollection[0][index2] == cLabelCollection[1][index]):
          correctlyCountedSeeds = correctlyCountedSeeds + 1
        
        # outgrageous number so the seed would not be considered as corresponding afterwards
        x_minIndexT[index2] =9999
        y_minIndexT[index2] = 9999
        x_maxIndexT[index2] = 9999
        y_maxIndexT[index2] = 9999

        break
    distanceCollection = []
    if(not seedCorrespondanceDetected):
      # seed correspondance not detected, use alt method of min distance where the seed closet to the one we are trying to find is the corresponding seed
      
      for index2 in range(numOfSeeds):    # we check each seed in the ori image 1 to find the seed we are looking for (from ori image 2)
        x_minIndexTIndex = x_minIndexT[index2] 
        y_minIndexTIndex = y_minIndexT[index2]
        x_maxIndexTIndex = x_maxIndexT[index2]
        y_maxIndexTIndex = y_maxIndexT[index2]
        
        # homo coord for the bounding boxes of ori image 1
        homoPos1 = (np.array([x_minIndexTIndex, y_minIndexTIndex, 1])).T
        homoPos2 = (np.array([x_maxIndexTIndex, y_maxIndexTIndex, 1])).T

        rotatedHomoPos1 = homoPos1.copy()
        rotatedHomoPos2 = homoPos2.copy()
        
        # perfrom the rotations, scaling on the ori bounding boxes based on what was perfomed previously on image1
        rotationMatrix2 = rotationMatrixCollection[0]
        for rotationMatrixIndex in rotationMatrix2:
          rotatedHomoPos1 = rotationMatrixIndex @ rotatedHomoPos1 
          rotatedHomoPos2 = rotationMatrixIndex @ rotatedHomoPos2 

        # Euclidean for the min, max coord of bounding boxes of the seeds in the ori image after rotation and scaling, then added with padding
        euclidianPos1 = [rotatedHomoPos1[0]/rotatedHomoPos1[2] + paddingImagesCollection[0][0], rotatedHomoPos1[1]/rotatedHomoPos1[2] + paddingImagesCollection[0][1]]
        euclidianPos2 = [rotatedHomoPos2[0]/rotatedHomoPos2[2] + paddingImagesCollection[0][0], rotatedHomoPos2[1]/rotatedHomoPos2[2] + paddingImagesCollection[0][1]]
        
        # Get the center for the seed in ori image 1
        xSeedCenter1Index = (euclidianPos1[0] + euclidianPos2[0])/2
        ySeedCenter1Index = (euclidianPos1[1] + euclidianPos2[1])/2

        # Get the euclidean distance between the center of seed of image 1 and image 2
        xDistance = math.pow((seedX2 - xSeedCenter1Index), 2)
        yDistance = math.pow((seedY2 - ySeedCenter1Index), 2)

        eucliDistance = math.sqrt(xDistance + yDistance)
        distanceCollection.append(eucliDistance)

      # get the seed with the min diff in distance
      indexLocation2 = distanceCollection.index(min(distanceCollection))
      
      # closet seed from image 1 is the corresponding seed for the one we are looking for in image 2
      image = cv2.rectangle(image, (x_minIndexT[indexLocation2], y_minIndexT[indexLocation2]), (x_maxIndexT[indexLocation2], y_maxIndexT[indexLocation2]), color, thickness)
      image = cv2.putText(image, str(index), (int((x_minIndexT[indexLocation2] + x_maxIndexT[indexLocation2])/2) - 40, int((y_minIndexT[indexLocation2] + y_maxIndexT[indexLocation2])/2)+ 40) , cv2.FONT_HERSHEY_SIMPLEX, 3, color2, 10, cv2.LINE_AA)

      # write coordinate for right, left, front, rear
      writeFile(x_minIndexT[indexLocation2], y_minIndexT[indexLocation2], x_maxIndexT[indexLocation2], y_maxIndexT[indexLocation2], bbobPath1, numOfSeeds)
      
      # check if correspondance is detected correctly
      if(cLabelCollection[0][indexLocation2] == cLabelCollection[1][index]):
          correctlyCountedSeeds = correctlyCountedSeeds + 1

     # outgrageous number so the seed would not be considered as corresponding afterwards
      x_minIndexT[indexLocation2] = 9999
      y_minIndexT[indexLocation2] = 9999
      x_maxIndexT[indexLocation2] = 9999
      y_maxIndexT[indexLocation2] = 9999

    image2 = cv2.rectangle(image2, (x_minIndexT2[index], y_minIndexT2[index]), (x_maxIndexT2[index], y_maxIndexT2[index]), color, thickness)
    image2 = cv2.putText(image2, str(index), (int(xCenterIndex2) - 40, int(yCenterIndex2)+ 40) , cv2.FONT_HERSHEY_SIMPLEX, 3, color2, 10, cv2.LINE_AA)
    writeFile(x_minIndexT2[index], y_minIndexT2[index], x_maxIndexT2[index], y_maxIndexT2[index], bbobPath2, numOfSeeds)
  
  accuracyDetectedSeed = (correctlyCountedSeeds/numOfSeeds)* 100
  print("Number of correctly detected seeds is: ")
  print(accuracyDetectedSeed)

  if(writeToFile):
    
    filename = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/correspondanceAccuracy.csv"

    with open(filename, 'a') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow([seedType, setNum, source_orientation, dest_orientation, accuracyDetectedSeed])

  return image, image2
