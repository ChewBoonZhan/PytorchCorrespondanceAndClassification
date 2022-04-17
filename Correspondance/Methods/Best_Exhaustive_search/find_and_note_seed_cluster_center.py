import cv2

def find_and_note_seed_cluster_center(image, boundingBoxCollection):
  image2 = image.copy()
  radius = 20
  color = (255,0,0)

  # Line thickness of 2 px
  thickness = 20

  (x_min, y_min, x_max, y_max) = boundingBoxCollection
  numberOfSeed = len(x_min)
  seedCenterCollectionX = []
  seedCenterCollectionY = []
  for index in range(numberOfSeed):
    xCenter = (x_min[index] + x_max[index])/2
    yCenter = (y_min[index] + y_max[index])/2
    seedCenterCollectionX.append(xCenter)
    seedCenterCollectionY.append(yCenter)

  averageX = int(sum(seedCenterCollectionX)/numberOfSeed)
  averageY = int(sum(seedCenterCollectionY)/numberOfSeed)

  image2 = cv2.circle(image2, (averageX,averageY), radius, color, thickness)

  return image2, averageX, averageY