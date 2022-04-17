## function to form corresponding labelled bounding boxes 
# Correspondance between 2 image, when provided with homography matrix.
# The output image pair should be saved

import cv2
import numpy as np

def form_corresponding_bounding_boxes(img1In, img2In, homographyMatrix, boundingBox1):
  
  img1 = img1In.copy() #src img
  img2 = img2In.copy() #dst img

  # Blue color in BGR
  color = (255, 0, 0)
  color2 = (250,218,94)
  thickness = 10

  #get bounding box info of src image
  x_min = boundingBox1["x_min"]
  y_min = boundingBox1["y_min"]
  y_max = boundingBox1["y_max"]
  x_max = boundingBox1["x_max"]

  #total number of seeds in the src image
  numberOfSeeds = x_max.shape[0]

  #for each seed in the src image
  for index in range(numberOfSeeds):

    #retrieve its bounding box coordinates
    x_minIndex1 = x_min[index]
    y_minIndex1 = y_min[index]
    y_maxIndex1 = y_max[index]
    x_maxIndex1 = x_max[index]

    start_point = (x_minIndex1, y_minIndex1)
    end_point = (x_maxIndex1, y_maxIndex1)

    #draw bounding box on around it based on the coordinates 
    img1 = cv2.rectangle(img1, start_point, end_point, color, thickness)

    #find the center point of the bounding box
    xCenter = int(abs(x_minIndex1 + x_maxIndex1)/2)-40
    yCenter = int(abs(y_minIndex1 + y_maxIndex1)/2)+40
    #add label (1,2,3...)
    img1 = cv2.putText(img1, str(index), (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, 4, color2, 10, cv2.LINE_AA)
    
    #convert coordinates to homogeneous form
    homoGenMatrix = np.transpose(np.array([x_minIndex1, y_minIndex1, 1]))
    homoGenMatrix2 = np.transpose(np.array([x_maxIndex1, y_maxIndex1, 1]))

    #put coordinates through homography matrix transformation x y -> x'y'
    outHomoGen = homographyMatrix @ homoGenMatrix
    outHomoGen2 = homographyMatrix @ homoGenMatrix2

    #retrieve x' y'
    x_min2 = int(outHomoGen[0]/outHomoGen[2])
    y_min2 = int(outHomoGen[1]/outHomoGen[2])
    x_max2 = int(outHomoGen2[0]/outHomoGen2[2])
    y_max2 = int(outHomoGen2[1]/outHomoGen2[2])

    start_point = (x_min2, y_min2)
    end_point = (x_max2, y_max2)

    #draw bounding box on dst image based on the x'y' tranformed from the src image
    img2 = cv2.rectangle(img2, start_point, end_point, color, thickness)

    xCenter = int(abs(x_min2 + x_max2)/2)-40
    yCenter = int(abs(y_min2 + y_max2)/2)+40

    # img2 = cv2.putText(img2,"1", (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, thickness, 255)
    # Using cv2.putText() method
    #add label
    img2 = cv2.putText(img2, str(index), (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, 4, color2, 10, cv2.LINE_AA)

    # break
  
  return (img1.copy(), img2.copy())