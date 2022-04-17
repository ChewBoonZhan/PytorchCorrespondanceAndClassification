import numpy as np
def transformBoundingBox(homography, bounding_box):
  (x_min, y_min, x_max, y_max) = bounding_box
  numSeeds = len(x_min)
  x_minNew = []
  y_minNew = []
  x_maxNew = []
  y_maxNew = []

  for index in range(numSeeds):
    xMinIndex = x_min[index]
    yMinIndex = y_min[index]

    xYMin = (np.array([xMinIndex, yMinIndex, 1])).T

    xYMinNew =  homography @ xYMin

    x_minNew.append(int(xYMinNew[0]/xYMinNew[2]))
    y_minNew.append(int(xYMinNew[1]/xYMinNew[2]))

    xMaxIndex = x_max[index]
    yMaxIndex = y_max[index]

    xYMax = (np.array([xMaxIndex, yMaxIndex, 1])).T

    xYMaxNew = homography @ xYMax

    x_maxNew.append(int(xYMaxNew[0]/xYMaxNew[2]))
    y_maxNew.append(int(xYMaxNew[1]/xYMaxNew[2]))

  x_minNew = np.array(x_minNew)
  y_minNew = np.array(y_minNew)
  x_maxNew = np.array(x_maxNew)
  y_maxNew = np.array(y_maxNew)

  return (x_minNew, y_minNew, x_maxNew, y_maxNew)