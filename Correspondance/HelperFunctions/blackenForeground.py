import numpy as np

# seedImageCollection - image that contains all the seeds. Must be original image as the one in Moodle
# seedType - "bad_seeds" or "good_seeds"
# imageViewAngle - "front", "top", "right", "rear", "left"
# setNumber - 1, 2, 3, 4...
def blackenForeground(seedImageCollection, seedType, imageViewAngle, setNumber):
  
  file_path = "BBOX_Record/" + seedType + "/set" + str(setNumber) + "/" + imageViewAngle + "/"  # TODO: Update file path
  df = pd.read_csv(file_path + "bbox_record.csv")
  x_min = np.array(df.iloc[:,1].values)
  y_min = np.array(df.iloc[:,2].values)
  x_max = np.array(df.iloc[:,3].values)
  y_max = np.array(df.iloc[:,4].values)

  imageMatrixArray = np.zeros(seedImageCollection.shape, dtype = int)

  numOfElement = x_min.shape[0]

  for index in range(numOfElement):
    imageMatrixArray[int(y_min[index]):int(y_max[index]+1),
                     int(x_min[index]):int(x_max[index]+1)
                    ] = 1

  # clone the array to prevent change to original image
  returnSeedImage = seedImageCollection.copy()


  returnSeedImage = np.multiply(returnSeedImage, imageMatrixArray)
  returnSeedImage = returnSeedImage.astype(np.uint8) ####

  return returnSeedImage