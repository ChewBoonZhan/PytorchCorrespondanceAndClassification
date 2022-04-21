import csv
import os

def createCorrespondanceAccuracyFile():
  csv_row = ["Seed Type", "Set Number", "Source Orientation", "Destination Orientation", "Accuracy"]
  filename = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/correspondanceAccuracy.csv"

  with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_row)
