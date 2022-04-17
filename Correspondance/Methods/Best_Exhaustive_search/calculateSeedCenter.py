def calculateSeedCenter(boundingBox):
  (x_min, y_min, x_max, y_max)= boundingBox #x and y min max bbox of 1 seed
  numSeeds = len(x_min)
  xCenter = []
  yCenter = []
  for index in range(numSeeds):
    xCenter.append((x_min[index] + x_max[index])/2)
    yCenter.append((y_min[index] + y_max[index])/2)

  return xCenter, yCenter