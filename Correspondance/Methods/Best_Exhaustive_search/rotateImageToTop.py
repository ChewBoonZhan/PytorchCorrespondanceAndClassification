import sys
import os

import numpy as np

sys.path.insert(0, os.getcwd())

from rotation_matrix import rotation_matrix

def rotateImageToTop(image, orientation):
  imageSize = image.shape

  # in degree.
  rotationAngle = 0.0085

  if(orientation == "top"):
    rotationAngle = 0

  T = np.array([[1, 0, -image.shape[1]/2], [0, 1, -image.shape[0]/2], [0, 0, 1]])
    
  R = rotation_matrix(image, rotationAngle, "x")

  H = np.linalg.inv(T) @ R @ T

  # white_image = (np.ones(image.shape, np.uint8) * 255).astype(np.uint8)
  
  # # use this to keep black border to see diff
  # image = cv2.warpPerspective(image, H, (imageSize[1], imageSize[0]), white_image, borderMode=cv2.BORDER_TRANSPARENT)

  return H
