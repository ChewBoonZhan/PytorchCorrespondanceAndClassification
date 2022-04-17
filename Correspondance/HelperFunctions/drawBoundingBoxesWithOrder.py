import cv2

def drawBoundingBoxesWithOrder(image, points, boundingBoxPath):
  color = (255,0,0)
  color2 = (250,218,94)
  thickness = 10
  (x_min, y_min, x_max, y_max) = readBoundingBoxCSV(boundingBoxPath)
  numSeeds  = len(points)
  for index in range(numSeeds):
    seedCoordinate = points[index]
    xPos = seedCoordinate[0]
    yPos = seedCoordinate[1]
    for index2 in range(numSeeds):
      xMinIndex = x_min[index2]
      yMinIndex = y_min[index2]
      xMaxIndex = x_max[index2]
      yMaxIndex = y_max[index2]
      if(xPos >= xMinIndex and xPos <= xMaxIndex and yPos >= yMinIndex and yPos <= yMaxIndex):
        # this is the right bounding box for the current bounding box center
        image = cv2.rectangle(image, (xMinIndex, yMinIndex), (xMaxIndex, yMaxIndex), color, thickness)
        image = cv2.putText(image, str(index), (xPos-50, yPos+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color2, 10, cv2.LINE_AA)
        break

  return image