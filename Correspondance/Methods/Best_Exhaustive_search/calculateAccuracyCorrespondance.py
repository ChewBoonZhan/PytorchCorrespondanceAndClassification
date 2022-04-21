import csv
import numpy as np
import pandas as pd
import os
def calculateAccuracyCorrespondance():
  filename = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/correspondanceAccuracy.csv"
  df = pd.read_csv(filename)
  accuracy = np.array(df.iloc[:,4].values)
  
  numSeeds = len(accuracy)
  totalAccuracy = 0.0
  for index in range(numSeeds):
    totalAccuracy = totalAccuracy + float(accuracy[index])

  averageAccuracy = totalAccuracy/numSeeds
  print(averageAccuracy)
  with open(filename, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Average Accuracy", "", "", "", averageAccuracy])