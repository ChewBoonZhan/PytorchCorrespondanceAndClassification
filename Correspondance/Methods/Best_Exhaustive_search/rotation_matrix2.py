import math
import numpy as np

## Compute rotation matrix, where image is:
# - Translated origin goes to center of image
# - Rotation about origin of image (0,0)
# - Translation of -width/2 for x and -height/2 for y

def rotation_matrix2(image, angle, axis:str):
  assert axis=='x' or axis=='y' or axis=='z', "axis must be either 'x', 'y', or 'z'"
  angle_radius = (angle * math.pi)/180
  if axis=='x':   
    R = np.array([[1,0,0],[0, math.cos(angle_radius), math.sin(-angle_radius)],[0, math.sin(angle_radius), math.cos(angle_radius)]])
  elif axis=='y':
    R = np.array([[math.cos(angle_radius), 0, math.sin(angle_radius)], [0, 1, 0], [math.sin(angle_radius), 0, math.cos(angle_radius)]])
  elif axis=='z':
    R = np.array([[math.cos(angle_radius), math.sin(-angle_radius), 0],[math.sin(angle_radius), math.cos(angle_radius), 0], [0, 0, 1]])

                        # width                    # height
  T = np.array([[1, 0, -image.shape[1]/2], [0, 1, -image.shape[0]/2], [0, 0, 1]])
  
  # (multiplicative) inverse of a matrix np.linal
  H =  np.linalg.inv(T) @ R @ T
  # H = T


  return H
