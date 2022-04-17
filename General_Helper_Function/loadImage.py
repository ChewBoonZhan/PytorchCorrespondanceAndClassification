import cv2
#seedType - "Bad_seeds" or "Good_seeds"
#imageViewAngle - "front", "top", "right", "rear", "left"
#setNumber - 1, 2, 3..
def loadImage(seedType, imageViewAngle, setNumber, original=True):
  if original:
    file_path = ("Multiview_jpg/" + seedType + "/Set"    # TODO: Update file path
                + str(setNumber) + "/" + imageViewAngle 
                + "_S" + str(setNumber) + ".jpg")
  else:
    file_path = ("SIFT_try/" + seedType + "/S"            # TODO: Update file path
               + str(setNumber) + "/" + imageViewAngle 
               + "_S" + str(setNumber) + ".jpg")
  image = cv2.imread(file_path)
  return image