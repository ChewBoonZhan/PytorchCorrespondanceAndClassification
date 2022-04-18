### Write File RearrangeBBox.csv

import csv

def writeFile(x_min, y_min, x_max, y_max, bbobPath, numOfSeeds):
  # name of csv file
  filename = bbobPath + "/RearrangeBBox.csv"

  # data rows of csv file 
  rows = [str(x_min), str(y_min), str(x_max), str(y_max)]

  with open(filename, 'a') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(rows)