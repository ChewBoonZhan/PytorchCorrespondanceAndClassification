from operator import indexOf
import pandas as pd
import numpy as np
import csv
import os

# crops surrounding area of the image, leaving only all the seeds at the center of the image
# seedImageCollection - image that contains all the seeds. Must be original image as the one in Moodle
# seedType - "bad_seeds" or "good_seeds"
# imageViewAngle - "front", "top", "right", "rear", "left"
# setNumber - 1, 2, 3, 4...
def cropSideNoise(seedImageCollection, seedType, imageViewAngle, setNumber):

  #retrieve bbox csv of original images
  csv_header = ["placeholder", "x_min", "y_min", "x_max", "y_max", "pad", "cLabel"]
  
  file_path = os.getcwd() + "/../Data/OriginalData/BBOX_Record/" + seedType + "/set" + str(setNumber) + "/" + imageViewAngle + "/"
  df = pd.read_csv(file_path + "bbox_record.csv")
  x_min = np.array(df.iloc[:,1].values)
  y_min = np.array(df.iloc[:,2].values)
  x_max = np.array(df.iloc[:,3].values)
  y_max = np.array(df.iloc[:,4].values)

  # retrieve their assigned corresponding labels
  c_label = np.array(df.iloc[:,6].values)

  imgShape = seedImageCollection.shape

  height = imgShape[0]
  width = imgShape[1] 

  numOfElement = x_min.shape[0]

  padding = 100

  # looks at x, min
  leftBound = min(x_min) -padding
  if(leftBound < 0):
    leftBound = 0

  # looks at x, max
  rightBound = max(x_max) + padding
  if(rightBound > width):
    rightBound = max(x_max)

  # looks at y, max
  bottomBound = max(y_max) + padding
  if(bottomBound > height):
    bottomBound = max(y_max)

  # looks at y, min
  topBound = min(y_min) - padding
  if(topBound <0):
    topBound = 0

  seedImageCollectionReturn = []

  #crop image
  imageSeedIndex = seedImageCollection[int(topBound):int(bottomBound + 1),
                              int(leftBound):int(rightBound+1)]


  ## generating new bounding box for cropped image

  topLeftXBefPad = leftBound
  topLeftYBefPad = topBound

  newX_min = []
  newY_min = []
  newX_max = []
  newY_max = []


  # calculate coordinate before padding
  for index in range(numOfElement):
    newX_min.append(x_min[index] - topLeftXBefPad)
    newY_min.append(y_min[index] - topLeftYBefPad)
    newX_max.append(x_max[index] - topLeftXBefPad)
    newY_max.append(y_max[index] - topLeftYBefPad)

  seedTypeWrite = seedType

  seedTypeWrite = seedTypeWrite.replace(seedTypeWrite[0], seedTypeWrite[0].upper(), 1)
  outPath = os.getcwd() + "/../Data/ProcessedData/SIFT_try/BBOX/" + seedTypeWrite + "/S" + str(setNumber) +"/" +  imageViewAngle + "/"  

  # Create the CSV file if it does not exist
  outcsv = os.path.join(outPath, 'bbox_record.csv')

  if not os.path.exists(outcsv):
    os.makedirs(outPath)
  #   with open(outcsv, 'w', newline='') as file:
  #     writer = csv.writer(file)

  # create row to be written to the csv file
  rowsCsv = []
  for index in range(numOfElement):
    startX = (newX_min[index]).astype("int")
    startY = (newY_min[index]).astype("int")
    endX = (newX_max[index]).astype("int")
    endY = (newY_max[index]).astype("int")

    # corresponding label
    cLabelIndex = (c_label[index]).astype("int")

    rowsCsv.append([1, startX, startY, endX, endY, 1, cLabelIndex])
  
  with open(outcsv, 'w', newline='', encoding='UTF8') as fileCsv:
    writer = csv.writer(fileCsv)
    writer.writerow(csv_header)


    writer.writerows(rowsCsv)


  return (imageSeedIndex)

if __name__ == '__main__':
  # called when runned from command prompt
  print("cropSideNoise")