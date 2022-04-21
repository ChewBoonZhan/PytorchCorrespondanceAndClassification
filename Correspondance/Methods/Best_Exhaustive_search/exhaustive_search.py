### Exhaustive search - Compute Rotations, Translation and Scaling

import cv2
import numpy as np

import sys
import os

import copy

sys.path.insert(0, os.getcwd())

from rotation_matrix import rotation_matrix
from calculateSeedCenter import calculateSeedCenter
from find_and_note_seed_cluster_center import find_and_note_seed_cluster_center
from rotateImageToTop import rotateImageToTop
from rotation_matrix2 import rotation_matrix2
from transformBoundingBox import transformBoundingBox
from get_num_seed_in_bounding_box import get_num_seed_in_bounding_box

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV
from readCLabelCSV import readCLabelCSV

def exhaustive_search(imagepath1, imagepath2, bbobPath1, bbobPath2, orientation1, orientation2):
  
  orientation = [orientation1, orientation2]

  image = cv2.imread(imagepath1)
  image2 = cv2.imread(imagepath2)

  imageCollection = [image, image2]

  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(bbobPath1)
  (x_min2, y_min2, x_max2, y_max2) = readBoundingBoxCSV(bbobPath2)

  cLabel = readCLabelCSV(bbobPath1)
  cLabel2 = readCLabelCSV(bbobPath2)

  cLabelCollection = [cLabel, cLabel2]
  
  # check if the number of seed detected is same or not
  numSeed1 = len(x_min)
  numSeed2 = len(x_min2)

  if(numSeed1!= numSeed2):
    print("==============================================")
    print("Number of seed detected is different for at:")
    print(imagepath1)
    print(imagepath2)
    print("==============================================")

  boundingBoxCollection = [[x_min, y_min, x_max, y_max], [x_min2, y_min2, x_max2, y_max2]]

  seedClusterCenterCollectionX = []
  seedClusterCenterCollectionY = []

  finalWidthCollection = []
  finalHeightCollection = []

  rotationMatrixCollection = []

  transformedBoundingBox = []

  transformedSeedCenterX = []
  transformedSeedCenterY = []

  imageWithCenter = []

  extremeX = []
  extremeY = []

  for index in range(2):

    rotationMatrixIndex = []
    orientationIndex = orientation[index]
    imageIndex = imageCollection[index]

    #retrieve bbox info of seed image (cropped noise)
    x_minIndex = boundingBoxCollection[index][0]
    y_minIndex = boundingBoxCollection[index][1]
    x_maxIndex = boundingBoxCollection[index][2]
    y_maxIndex = boundingBoxCollection[index][3]

    finalWidth1 = 0
    finalHeight1 = 0
    R1 = 0

    #compute rotation matrix for adjusting orientation of image the same as top view
    #top view aligns with front view

    if(orientationIndex == "front"):
      # no rotation required
      finalWidth1 = imageIndex.shape[1]
      finalHeight1 = imageIndex.shape[0]
      R1 = rotation_matrix(imageIndex, 0, "z") 
      
    elif(orientationIndex == "left"):
      # rotate 90 anticlockwise
      R1 = rotation_matrix(imageIndex, 90, "z")
      finalWidth1 = imageIndex.shape[0]
      finalHeight1 = imageIndex.shape[1]

    elif(orientationIndex == "right"):
      # rotate 90 clockwise about z-axis
      R1 = rotation_matrix(imageIndex, -90, "z")
      finalWidth1 = imageIndex.shape[0]
      finalHeight1 = imageIndex.shape[1]
      
      
    elif(orientationIndex == "top" or orientationIndex == "rear"):
      # rotate 180 degrees
      R1 = rotation_matrix(imageIndex, 180, "z")
      finalWidth1 = imageIndex.shape[1]
      finalHeight1 = imageIndex.shape[0]      
    
    #record the width and height of image
    finalWidthCollection.append(finalWidth1)
    finalHeightCollection.append(finalHeight1)

    #compute a rotational matrix to make the image not slant at 45 degree
    H2 = rotateImageToTop(imageIndex, orientationIndex)

    #record the 2 rotation matrices : H2 and R1
    rotationMatrixIndex.append(H2) 
    rotationMatrixIndex.append(R1) 

    #apply rotation matrices on bbox coordinates, H2 then R1
    (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT) = transformBoundingBox(H2, (x_minIndex, y_minIndex, x_maxIndex, y_maxIndex))
    (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT) = transformBoundingBox(R1, (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT))

    #new bbox coordinates after the rotation transformation
    transformedBoundingBox.append([x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT])

    #calculate the centers of each seed based on their transformed bbox
    #each bbox will have a center (x,y)
    xCenterIndexCollection, yCenterIndexCollection = calculateSeedCenter((x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT))

    #record the center x and y coordinates of each seed
    transformedSeedCenterX.append(xCenterIndexCollection) #[[x of bbox seed1, x of bbox seed2,...]]
    transformedSeedCenterY.append(yCenterIndexCollection) #[[y of bbox seed1, y of bbox seed2,...]]

    #find the centers with min and max x
    minXOfImage = min(xCenterIndexCollection)
    maxXOfImage = max(xCenterIndexCollection)

    #find the centers with min and max y
    minYOfImage = min(yCenterIndexCollection)
    maxYOfImage = max(yCenterIndexCollection)

    #record 
    #extremeX = [[seed center with minx of image1, seed center with maxX of image1], [seed center with minX of image2, seed center with maxX of image2]]
    extremeX.append([minXOfImage, maxXOfImage])
    extremeY.append([minYOfImage, maxYOfImage])

    white_image = (np.ones(image.shape, np.uint8) * 255).astype(np.uint8)

    #warp image based on the rotation matrices -- H2 then R1
    imageIndex = cv2.warpPerspective(imageIndex, H2, (imageIndex.shape[1], imageIndex.shape[0]), white_image, borderMode=cv2.BORDER_TRANSPARENT)
    imageIndex = cv2.warpPerspective(imageIndex, R1, (int(finalWidth1), int(finalHeight1)), white_image, borderMode=cv2.BORDER_TRANSPARENT)

    #find the center point (x,y) among ALL seeds using the transformed bbox coordinates of each seed and the warped image
    imageSeedCenter, seedCenterX, seedCenterY = find_and_note_seed_cluster_center(imageIndex, (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT))
    imageWithCenter.append(imageSeedCenter) #image with the center point shown

    #record the center point
    seedClusterCenterCollectionX.append(seedCenterX)
    seedClusterCenterCollectionY.append(seedCenterY)

    #append the seed rotation matrices H2 and R1
    rotationMatrixCollection.append(rotationMatrixIndex) #[[H2_img1,R1_img1], [H2_img2,R1_img2]]


  ##############################################################################
  # scale both images to make their min & max of x and y having the same distance
  
  #seed centers with min x and max x of image 1
  minXImage1 = extremeX[0][0]
  maxXImage1 = extremeX[0][1]

  #the x difference between the seed centers
  diffXImage1 = maxXImage1 - minXImage1 

  #seed centers with min y and max y of image 1
  minYImage1 = extremeY[0][0]
  maxYImage1 = extremeY[0][1]

  #the y difference between the seed centers
  diffYImage1 = maxYImage1 - minYImage1

  minXImage2 = extremeX[1][0]
  maxXImage2 = extremeX[1][1]

  diffXImage2 = maxXImage2 - minXImage2

  minYImage2 = extremeY[1][0]
  maxYImage2 = extremeY[1][1]

  diffYImage2 = maxYImage2 - minYImage2

  #find the ratio difference of x and y distance between the 2 images
  scaleX = diffXImage1/diffXImage2
  scaleY = diffYImage1/diffYImage2

  #compute a scaling matrix
  T = np.array([[scaleX, 0, 0], [0, scaleY, 0], [0, 0, 1]])

  white_image = (np.ones(image.shape, np.uint8) * 255).astype(np.uint8)

  #scale 2nd image so that the x and y distance of its seed centers are the same as the 1st image
  imageWithCenter[1] = cv2.warpPerspective(imageWithCenter[1] , T, (int(imageWithCenter[1] .shape[1]*scaleX), int(imageWithCenter[1] .shape[0]*scaleY)), white_image, borderMode=cv2.BORDER_TRANSPARENT)

  #update the bounding box of the 2nd image
  x_minIndexT = transformedBoundingBox[1][0]
  y_minIndexT = transformedBoundingBox[1][1]
  x_maxIndexT = transformedBoundingBox[1][2]
  y_maxIndexT = transformedBoundingBox[1][3]

  (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT) = transformBoundingBox(T, (x_minIndexT, y_minIndexT, x_maxIndexT, y_maxIndexT))

  transformedBoundingBox[1][0] = x_minIndexT
  transformedBoundingBox[1][1] = y_minIndexT
  transformedBoundingBox[1][2] = x_maxIndexT
  transformedBoundingBox[1][3] = y_maxIndexT

  #update the center point (x,y) of the 2nd image
  xUpdate = seedClusterCenterCollectionX[1]
  yUpdate = seedClusterCenterCollectionY[1]

  homoCoord = (np.array([xUpdate, yUpdate, 1])).T 
  homoNew =T @ homoCoord 

  newXUpdate = homoNew[0]/homoNew[2] 
  newYUpdate = homoNew[1]/homoNew[2]

  seedClusterCenterCollectionX[1] =newXUpdate 
  seedClusterCenterCollectionY[1] = newYUpdate

  #update the transformation matrix
  rotationMatrixCollection[1].append(T)


  ###############################################################################
  ## Add padding to image to make seed center to correct location, so both seed center lie on top of each other

  imageCenter1 = imageWithCenter[0]
  imageCenter2 = imageWithCenter[1]

  #imageCenter2Height = imageCenter2.shape[0]
  #imageCenter2Width = imageCenter2.shape[1]

  #center point of 2nd image - center point of 1st image
  diffInX = seedClusterCenterCollectionX[1] - seedClusterCenterCollectionX[0] + 0.0
  diffInY = seedClusterCenterCollectionY[1] - seedClusterCenterCollectionY [0] + 0.0

  paddingImagesCollection = []
  paddingImage1 = []
  paddingImage2 = []

  if(diffInX >0): #2nd image center point at x > 1st image
    # add padding instead to image 1
    imageCenter1 = cv2.copyMakeBorder(imageCenter1, 0, 0, int(abs(diffInX)), 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
    
    # update all the seed bounding box and center..
    seedClusterCenterCollectionX[0] = seedClusterCenterCollectionX[0] + int(abs(diffInX))

    # TODO: Update bounding box here
    transformedBoundingBox[0][0]= transformedBoundingBox[0][0]+ int(abs(diffInX))
    transformedBoundingBox[0][2]= transformedBoundingBox[0][2]+ int(abs(diffInX))
    
    paddingImage1.append(int(abs(diffInX)))
    paddingImage2.append(0)
  
  else:
    imageCenter2 = cv2.copyMakeBorder(imageCenter2, 0, 0, int(abs(diffInX)), 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

    # update all the seed bounding box and center..
    seedClusterCenterCollectionX[1] = seedClusterCenterCollectionX[1] + int(abs(diffInX))

    # TODO: Update bounding box here
    transformedBoundingBox[1][0]= transformedBoundingBox[1][0] + int(abs(diffInX))
    transformedBoundingBox[1][2]= transformedBoundingBox[1][2] + int(abs(diffInX))
    
    paddingImage1.append(0)
    paddingImage2.append(int(abs(diffInX)))
    
  if(diffInY > 0): #2nd image center point at y > 1st image
    # add padding instead to image 1
    imageCenter1 = cv2.copyMakeBorder(imageCenter1, int(abs(diffInY)), 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

    # update all seed bounding box and center...
    seedClusterCenterCollectionY[0] = seedClusterCenterCollectionY[0] + int(abs(diffInY))

    # TODO: Update bounding box here
    transformedBoundingBox[0][1]= transformedBoundingBox[0][1]+ int(abs(diffInY))
    transformedBoundingBox[0][3]= transformedBoundingBox[0][3]+ int(abs(diffInY))

    paddingImage1.append(int(abs(diffInY)))
    paddingImage2.append(0)

  else:
    imageCenter2 = cv2.copyMakeBorder(imageCenter2, int(abs(diffInY)), 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

    # update all the seed bounding box and center..
    seedClusterCenterCollectionY[1] = seedClusterCenterCollectionY[1] + int(abs(diffInY))
    
    # TODO: Update bounding box here

    transformedBoundingBox[1][1]= transformedBoundingBox[1][1] + int(abs(diffInY))
    transformedBoundingBox[1][3]= transformedBoundingBox[1][3]+ int(abs(diffInY))
    
    paddingImage1.append(0)
    paddingImage2.append(int(abs(diffInY)))
  
  #record how much have been padded in both x and y directions
  paddingImagesCollection.append(paddingImage1) 
  paddingImagesCollection.append(paddingImage2)
  

  ################################################################################
  # pad image 1 or 2 according to their size, to make both image same size
  imageCenter1Height = imageCenter1.shape[0]
  imageCenter1Width = imageCenter1.shape[1]

  imageCenter2Height = imageCenter2.shape[0]
  imageCenter2Width = imageCenter2.shape[1]

  diffHeight = int(abs(imageCenter1Height - (imageCenter2Height)))
  
  if(imageCenter1Height < (imageCenter2Height)):
    # image 1 is smaller than image 2
    # pad image 1
    imageCenter1 = cv2.copyMakeBorder(imageCenter1, 0, diffHeight, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
  
  else:
    #pad image 2
    imageCenter2 = cv2.copyMakeBorder(imageCenter2, 0, diffHeight, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
  
  diffWidth = int(abs(imageCenter1Width - (imageCenter2Width)))

  if(imageCenter1Width < (imageCenter2Width)):
    # image 1 is smaller than image 2
    # pad image 1
    imageCenter1 = cv2.copyMakeBorder(imageCenter1, 0, 0, 0, diffWidth, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

  else:
    #pad image 2
    imageCenter2 = cv2.copyMakeBorder(imageCenter2, 0, 0, 0, diffWidth, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

  ###################################################################################
  # translate the seed center to the center of image, then rotate the second image

  # seed center
  xPosCenter = seedClusterCenterCollectionX[1]
  yPosCenter = seedClusterCenterCollectionY[1]

  imageTrialRotate = imageCenter2.copy()

  xCenterImagePos = int(imageTrialRotate.shape[1]/2)
  yCenterImagePos = int(imageTrialRotate.shape[0]/2)

  translateX = abs(xCenterImagePos - xPosCenter)
  translateY = abs(yCenterImagePos - yPosCenter)

  # to make seed average center center of image
  T2 = np.array([[1, 0, translateX], [0, 1, translateY], [0, 0, 1]])

  white_image = (np.ones(imageTrialRotate.shape, np.uint8) * 255).astype(np.uint8)

  maxSeedDetected = 0
  rotMatrixToUse = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

  for rotationIndex in range(360):
    imageTrialRotateIndex = imageTrialRotate.copy()
    R = rotation_matrix2(imageTrialRotate, rotationIndex, "z")
    
    T = T2 @ R @ np.linalg.inv(T2)

    transformedImage = cv2.warpPerspective(imageTrialRotateIndex, T, (imageTrialRotate.shape[1], imageTrialRotate.shape[0]), white_image, borderMode=cv2.BORDER_TRANSPARENT)

    rotationMatrixCollectionClone = copy.deepcopy(rotationMatrixCollection)
    
    rotationMatrixCollectionClone[1].append(T)

    # copy.deepcopy copies the entire list 
    # values changed wont affect original list
    foundSeed, seedNum = get_num_seed_in_bounding_box(copy.deepcopy(boundingBoxCollection), rotationMatrixCollectionClone, paddingImagesCollection)
    
    if(foundSeed == seedNum):
      maxSeedDetected = foundSeed
      # all the seed are in bounding box, proceed
      rotMatrixToUse = T
      break
    else:
      if(foundSeed >maxSeedDetected):
        maxSeedDetected = foundSeed
        rotMatrixToUse = T

  imageTrialRotateIndex = imageTrialRotate.copy()
  transformedImage = cv2.warpPerspective(imageTrialRotateIndex, rotMatrixToUse, (imageTrialRotate.shape[1], imageTrialRotate.shape[0]), white_image, borderMode=cv2.BORDER_TRANSPARENT)

  rotationMatrixCollection[1].append(rotMatrixToUse)

  # tempBoundingBoxDraw
  imageOut = (0.5 * imageCenter1 + 0.5 * transformedImage).astype(np.uint8)
  imageOut2 = (0.5 * imageCenter1 + 0.5 * imageCenter2).astype(np.uint8)

  # f, axarr = plt.subplots(2)
  # axarr[0].imshow(imageOut)
  # axarr[1].imshow(imageOut2)
  

  return image, image2, boundingBoxCollection, transformedBoundingBox, rotationMatrixCollection, paddingImagesCollection, cLabelCollection

  

# function note: bounding box is not updated after images are rotated, translated, scaled

