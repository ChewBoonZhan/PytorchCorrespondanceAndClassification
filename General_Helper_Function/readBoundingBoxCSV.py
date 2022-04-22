import pandas as pd
import numpy as np

def readBoundingBoxCSV(file_path, rearrange=False):
  if not(rearrange):
    df = pd.read_csv(file_path + "bbox_record.csv")
    x_min = np.array(df.iloc[:,1].values)
    y_min = np.array(df.iloc[:,2].values)
    x_max = np.array(df.iloc[:,3].values)
    y_max = np.array(df.iloc[:,4].values)
  else:
    df = pd.read_csv(file_path + "RearrangeBBox.csv")
    x_min = np.array(df.iloc[:,0].values)
    y_min = np.array(df.iloc[:,1].values)
    x_max = np.array(df.iloc[:,2].values)
    y_max = np.array(df.iloc[:,3].values)

  return (x_min, y_min, x_max, y_max)