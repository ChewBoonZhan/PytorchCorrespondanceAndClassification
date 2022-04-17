### Find the number of seed center 2 that is in the bounding box of image 1. 

# the bbox are the bbox of cropped images only

# transformation matrices and padding info are obtained from the above computation (exhaustive search function)

import numpy as np


def get_num_seed_in_bounding_box(boundingBoxCollectionIn, rotationMatrixCollection, paddingImagesCollection):
  boundingBoxCollectionHere = boundingBoxCollectionIn
  color = (255, 0, 0)
  color2 = (250,218,94)
  thickness = 10

  # seed center from image 2
  boundingBox2 = boundingBoxCollectionHere[1]
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
  boundingBox1 = boundingBoxCollectionHere[0]
  x_minIndexT = boundingBox1[0]
  y_minIndexT = boundingBox1[1]
  x_maxIndexT = boundingBox1[2]
  y_maxIndexT = boundingBox1[3]
  
  numSeedsGot = 0

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


        # outgrageous number so the seed would not be considered as corresponding afterwards
        x_minIndexT[index2] =9999
        y_minIndexT[index2] = 9999
        x_maxIndexT[index2] = 9999
        y_maxIndexT[index2] = 9999

        numSeedsGot = numSeedsGot + 1

        break
    
  return numSeedsGot,numOfSeeds
