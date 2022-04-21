import pandas as pd
import numpy as np
def readCLabelCSV(file_path):
  df = pd.read_csv(file_path + "bbox_record.csv")
  c_label = np.array(df.iloc[:,6].values)
    

  return c_label