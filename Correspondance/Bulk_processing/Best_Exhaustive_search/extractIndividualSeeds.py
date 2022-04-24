import os
import csv
import cv2
import sys

from extractSeeds import extractSeeds
from loadImageExtractSeed import loadImage

#this method will extract individual corresponding seeds (from all 5 views) and save them together in a folder
def extract_seeds():

  pathTrain = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Training'
  isExist = os.path.exists(pathTrain)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTrain)

  pathTest = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Testing'
  isExist = os.path.exists(pathTest)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTest)

  pathTrainGood = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Training/Good_seeds'
  isExist = os.path.exists(pathTrainGood)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTrainGood)

  pathTrainBad = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Training/Bad_seeds'
  isExist = os.path.exists(pathTrainBad)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTrainBad)

  pathTestGood = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Testing/Good_seeds'
  isExist = os.path.exists(pathTestGood)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTestGood)

  pathTestBad = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Testing/Bad_seeds'
  isExist = os.path.exists(pathTestBad)
  #if directory doesnt exist
  if not isExist: 
    # Create a new directory 
    os.makedirs(pathTestBad)

  #good seeds
  for index in range(10):
    if index < 8:
      path = pathTrainGood + '/S'+ str(index+1) # example /SIFT_try/Training/Good_seeds/S1
    else:
      path = pathTestGood + '/S'+ str(index+1) # example /SIFT_try/Training/Good_seeds/S1
    
    isExist = os.path.exists(path)
    #if directory doesnt exist
    if not isExist: 
      # Create a new directory 
      os.makedirs(path)

    print("Cropping good seeds Set" + str(index+1) + "...")

    #load set
    image_top = loadImage("Good_seeds", "top", index+1, False)
    image_right = loadImage("Good_seeds", "right", index+1, False)
    image_left = loadImage("Good_seeds", "left", index+1, False)
    image_front = loadImage("Good_seeds", "front", index+1, False)
    image_rear = loadImage("Good_seeds", "rear", index+1, False)

    #extract individual seeds from the set
    seedImageCollection_top = (extractSeeds(image_top, "Good_seeds", "top", index+1, False))
    seedImageCollection_right = (extractSeeds(image_right, "Good_seeds", "right", index+1, False))
    seedImageCollection_left = (extractSeeds(image_left, "Good_seeds", "left", index+1, False))
    seedImageCollection_front = (extractSeeds(image_front, "Good_seeds", "front", index+1, False))
    seedImageCollection_rear = (extractSeeds(image_rear, "Good_seeds", "rear", index+1, False))

    #save each individual seed
    i = 0
    for index_seed in range(seedImageCollection_top.shape[0]):
      i = i + 1
      if index < 8:
        path = pathTrainGood + '/S'+ str(index+1) + '/Seed' + str(i) # example /SIFT_try/Training/Good_seeds/S1/Seed1
      else:
        path = pathTestGood + '/S'+ str(index+1) + '/Seed' + str(i) # example /SIFT_try/Testing/Good_seeds/S1/Seed1
      isExist = os.path.exists(path)
      #if directory doesnt exist
      if not isExist: 
        # Create a new directory 
        os.makedirs(path)
      
      cv2.imwrite(path + "/top.jpg", seedImageCollection_top[index_seed])
      cv2.imwrite(path + "/right.jpg", seedImageCollection_right[index_seed])
      cv2.imwrite(path + "/left.jpg", seedImageCollection_left[index_seed])
      cv2.imwrite(path + "/front.jpg", seedImageCollection_front[index_seed])
      cv2.imwrite(path + "/rear.jpg", seedImageCollection_rear[index_seed])

  #bad seeds
  for index in range(12):
    if index < 9:
      path = pathTrainBad + '/S'+ str(index+1) # example /SIFT_try/Training/Bad_seeds/S1
    else:
      path = pathTestBad + '/S'+ str(index+1) # example /SIFT_try/Training/Bad_seeds/S1
    isExist = os.path.exists(path)
    #if directory doesnt exist
    if not isExist: 
      # Create a new directory 
      os.makedirs(path)

    print("Cropping bad seeds Set" + str(index+1) + "...")
    #load set
    image_top = loadImage("Bad_seeds", "top", index+1, False)
    image_right = loadImage("Bad_seeds", "right", index+1, False)
    image_left = loadImage("Bad_seeds", "left", index+1, False)
    image_front = loadImage("Bad_seeds", "front", index+1, False)
    image_rear = loadImage("Bad_seeds", "rear", index+1, False)

    #extract individual seeds from the set
    seedImageCollection_top = (extractSeeds(image_top, "Bad_seeds", "top", index+1, False))
    seedImageCollection_right = (extractSeeds(image_right, "Bad_seeds", "right", index+1, False))
    seedImageCollection_left = (extractSeeds(image_left, "Bad_seeds", "left", index+1, False))
    seedImageCollection_front = (extractSeeds(image_front, "Bad_seeds", "front", index+1, False))
    seedImageCollection_rear = (extractSeeds(image_rear, "Bad_seeds", "rear", index+1, False))

    #save each individual seed
    i = 0
    for index_seed in range(seedImageCollection_top.shape[0]):
      i = i + 1
      if index < 9:
        path = pathTrainBad + '/S'+ str(index+1) + '/Seed' + str(i) # example /SIFT_try/Training/Bad_seeds/S1/Seed1
      else:
        path = pathTestBad + '/S'+ str(index+1) + '/Seed' + str(i) # example /SIFT_try/Testing/Bad_seeds/S1/Seed1
      isExist = os.path.exists(path)
      #if directory doesnt exist
      if not isExist: 
        # Create a new directory 
        os.makedirs(path)
      cv2.imwrite(path + "/top.jpg", seedImageCollection_top[index_seed])
      cv2.imwrite(path + "/right.jpg", seedImageCollection_right[index_seed])
      cv2.imwrite(path + "/left.jpg", seedImageCollection_left[index_seed])
      cv2.imwrite(path + "/front.jpg", seedImageCollection_front[index_seed])
      cv2.imwrite(path + "/rear.jpg", seedImageCollection_rear[index_seed])


if __name__ == '__main__':

  extract_seeds()