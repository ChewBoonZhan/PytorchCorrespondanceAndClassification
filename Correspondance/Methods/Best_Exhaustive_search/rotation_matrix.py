import math
import numpy as np

# Define a function to determine the rotational matrix based on the angle rotated and the axis rotated about
# related to rotational matrix
# rotating them to the top view, regardless of the orientation

def rotation_matrix(image, angle, axis:str):
  assert axis=='x' or axis=='y' or axis=='z', "axis must be either 'x', 'y', or 'z'"
  angle_radius = (angle * math.pi)/180
  if axis=='x':   
    R = np.array([[1,0,0],[0, math.cos(angle_radius), math.sin(-angle_radius)],[0, math.sin(angle_radius), math.cos(angle_radius)]])
  elif axis=='y':
    R = np.array([[math.cos(angle_radius), 0, math.sin(angle_radius)], [0, 1, 0], [math.sin(angle_radius), 0, math.cos(angle_radius)]])
  elif axis=='z':
    R = np.array([[math.cos(angle_radius), math.sin(-angle_radius), 0],[math.sin(angle_radius), math.cos(angle_radius), 0], [0, 0, 1]])

  if(angle == -90):
    # move downwards by width of image
    T2 = np.array([[1, 0, 0], [0, 1, image.shape[1]], [0, 0, 1]])
  elif(angle == 90):
    # move rightward by height of image
    T2 = np.array([[1, 0, image.shape[0]], [0, 1, 0], [0, 0, 1]])
  elif(angle == 180):
    # move rightward, and downward
    # T2 = np.array([[1, 0, image.shape[0]], [0, 1, image.shape[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, image.shape[1]], [0, 1, image.shape[0]], [0, 0, 1]])
  else:
    T2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

  # T2 = np.array([[1, 0, 500], [0, 1, 0], [0, 0, 1]])

  # H = np.linalg.inv(T2) @ R @ T2 

  H =  T2 @ R


  return H
