###Create File RearrangeBBox.csv
import csv
def createFile(bbobPath):
  filename = bbobPath + "/RearrangeBBox.csv"

  # field names 
  fields = ['x_min','y_min','x_max','y_max']

  # writing to csv file
  with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)