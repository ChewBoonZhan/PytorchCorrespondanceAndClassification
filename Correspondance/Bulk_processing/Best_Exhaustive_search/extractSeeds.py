from matplotlib import image
import pandas as pd
import numpy as np

import sys
import os

import cv2

# seedImageCollection - image that contains all the seeds. Must be original image as the one in Moodle
# seedType - "bad_seeds" or "good_seeds"
# imageViewAngle - "front", "top", "right", "rear", "left"
# setNumber - 1, 2, 3, 4...
def extractSeeds(seedImageCollection, seedType, imageViewAngle, setNumber, original=True):
  imageParam = seedImageCollection.shape
  imageHeight = imageParam[0]
  imageWidth = imageParam[1]
  
  if original:
    file_path = os.getcwd() + "/../../../Data/OriginalData/BBOX_Record/" + seedType + "/set" + str(setNumber) + "/" + imageViewAngle + "/bbox_record.csv"   
    df = pd.read_csv(file_path)
    x_min = np.array(df.iloc[:,1].values)
    y_min = np.array(df.iloc[:,2].values)
    x_max = np.array(df.iloc[:,3].values)
    y_max = np.array(df.iloc[:,4].values)
  else:
    file_path = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/BBOX/" + seedType + "/S" + str(setNumber) + "/" + imageViewAngle + "/RearrangeBBox.csv"  
    df = pd.read_csv(file_path)
    x_min = np.array(df.iloc[:,0].values)
    y_min = np.array(df.iloc[:,1].values)
    x_max = np.array(df.iloc[:,2].values)
    y_max = np.array(df.iloc[:,3].values)

  #changing cropped image dimension (...x...)
  finalDim = 400
  # paddingLess = np.full(coordinateShape, 100)

  numOfElement = x_min.shape[0]

  seedImageCollectionReturn = []

  for index in range(numOfElement):
    imageSeedIndex = seedImageCollection[int(y_min[index]):int(y_max[index]+1),
                            int(x_min[index]):int(x_max[index]+1)]
    seedIndexHeight = abs(y_min[index] - y_max[index])
    seedIndexWidth = abs(x_min[index] - x_max[index])


    if(seedIndexHeight < seedIndexWidth):
      # height < width
      widthResize = finalDim
      heightResize = int((widthResize/seedIndexWidth) *seedIndexHeight )

    else:
      # width < height
      heightResize = finalDim
      widthResize = int((heightResize/seedIndexHeight) * seedIndexWidth)

    imageSeedIndex = cv2.resize(imageSeedIndex, (widthResize, heightResize), interpolation = cv2.INTER_NEAREST)


    differencePaddingY = finalDim - heightResize
    differencePaddingX = finalDim - widthResize


    topPad  = int(differencePaddingY/2)
    bottomPad = int(differencePaddingY/2)

    sumHeight = topPad + bottomPad + heightResize
   
    # sanity check to make sure height is consistent
    if(sumHeight <finalDim):
      # find the difference
      diff = finalDim - sumHeight
      
      # add it into topPad
      topPad = topPad + diff
    elif(sumHeight >finalDim):
      # find the difference
      diff = sumHeight - finalDim

      #minus it out from topPad
      topPad = topPad - diff

    leftPad = int(differencePaddingX/2)
    rightPad = int(differencePaddingX/2)

    sumWidth = leftPad + rightPad + widthResize


    # sanity check to make sure width is consistent
    if(sumWidth <finalDim):
      # find the difference
      diff = finalDim - sumWidth
      
      # add it into topPad
      leftPad = leftPad + diff
    elif(sumWidth > finalDim):
      # find the difference
      diff = sumWidth - finalDim

      #minus it out from topPad
      leftPad = leftPad - diff




    whiteColor = (255,255,255)
    imageSeedIndex = cv2.copyMakeBorder(imageSeedIndex,topPad,bottomPad,leftPad,rightPad,cv2.BORDER_CONSTANT,value=whiteColor)

    seedImageCollectionReturn.append(imageSeedIndex)

  
  return np.array(seedImageCollectionReturn)

if __name__ == '__main__':  # Stopped here
  print(os.path. exists(os.getcwd() + "/../../../Data/OriginalData/BBOX_Record/"))
  print(os.path. exists(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/"))
  