import cv2
import os
#seedType - "Bad_seeds" or "Good_seeds"
#imageViewAngle - "front", "top", "right", "rear", "left"
#setNumber - 1, 2, 3..
def loadImage(seedType, imageViewAngle, setNumber, original=True):
  if original:
    file_path = (os.getcwd() + "/../../../Data/OriginalData/Multiview_jpg/" + seedType + "/Set"    
                + str(setNumber) + "/" + imageViewAngle 
                + "_S" + str(setNumber) + ".jpg")
  else:
    file_path = (os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/" + seedType + "/S"            
               + str(setNumber) + "/" + imageViewAngle 
               + "_S" + str(setNumber) + ".jpg")
  image = cv2.imread(file_path)
  return image


if __name__ == '__main__':
  print(os.path. exists(os.getcwd() + "/../../../Data/OriginalData/Multiview_jpg/"))
  print(os.path. exists(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/"))