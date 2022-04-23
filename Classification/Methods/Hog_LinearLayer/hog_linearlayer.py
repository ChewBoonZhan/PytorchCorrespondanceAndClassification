import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from dataset import loadTestData, loadTrainData

import cv2
import numpy as np
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from skimage.feature import hog
from sklearn.metrics import classification_report
import csv
import pandas as pd

from save_results_csv import save_results_csv
from save_results_corr_image import save_results_corr_image

from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_image_size = 128
batch_size = 8

trainDataHog = []
testDataHog = []
testFilePaths= []
def hog_linearlayer():

   global trainDataHog, testDataHog, testFilePaths, testData

   #load training and testing datasets
   trainData = loadTrainData()
   testData = loadTestData()
   
   #retrieve labels
   trainLabel = []
   for i in range(len(trainData)):
       trainLabel.append(trainData[i][0][1])

   testLabel = []
   for i in range(len(testData)):
       testLabel.append(testData[i][0][1])

   #extract hog of training
   print("\nExtract Hog Features from Training datasets...")
   for trainDataIndex in range(len(trainData)):
        trainPathCollection = trainData[trainDataIndex]
        trainDataHogIndex = []
        for trainPath in trainPathCollection:
            image = cv2.imread(trainPath[0]) 
            image = (cv2.resize(image, (input_image_size, input_image_size)))/255
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
            trainDataHogIndex = trainDataHogIndex + (fd.tolist())
            
        trainDataHog.append(trainDataHogIndex)
   
   #extract hog of testing
   print("\nExtract Hog Features from Testing datasets...")
   for testDataIndex in range(len(testData)):
       testPathCollection = testData[testDataIndex]
       testDataHogIndex = []
       testFilePaths.append(testPathCollection[0][0])
       for testPath in testPathCollection:
           image = cv2.imread(testPath[0]) 
           image = (cv2.resize(image, (input_image_size, input_image_size)))/255
           fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
           testDataHogIndex = testDataHogIndex + (fd.tolist())

       testDataHog.append(testDataHogIndex)
       
   testDataHog = np.array(testDataHog)

   #load to tensor
   print("\nLoading to Tensor...")
   train_loader = tensor_train(trainDataHog,trainLabel)
   tensorTestData = tensor_test(testDataHog,testLabel)
   
   #run on classifier model
   run_linearlayermodel(train_loader, tensorTestData)



def tensor_train(trainDataHog,trainLabel):

   #load training hog to tensor
   trainDataHog = torch.tensor(np.array(trainDataHog))
   trainLabel = torch.tensor(trainLabel)
   # creating training tensor dataset
   tensorTrainData = TensorDataset(trainDataHog, trainLabel)
   # creating data loader for data
   train_loader = DataLoader(tensorTrainData, batch_size, shuffle=True)
   # send data loaded to device
   train_loader = DeviceDataLoader(train_loader, device)

   return train_loader


def tensor_test(testDataHog,testLabel):

   #load testing hog to tensor
   testDataHog = torch.tensor(testDataHog)
   testLabel = torch.tensor(testLabel)
   # creating testing tensor dataset
   tensorTestData = TensorDataset(testDataHog, testLabel)
   tensorTestData = DeviceDataLoader(tensorTestData, device)

   return tensorTestData


def run_linearlayermodel(train_loader, tensorTestData):

    linearModel = LinearClassifier().to(device)   
    linearModelOptimizer = torch.optim.SGD(linearModel.parameters(), lr=0.001, momentum=0.9)
    loss = torch.nn.BCEWithLogitsLoss()

    learning_rate = 0.001
    num_epoch = 50

    if(os.path.exists("linearHOG.pth") ):
        # load the model
        print("\Loading existing model...")
        linearModel.load_state_dict(torch.load("linearHOG.pth", map_location=device))
    else:
        # pretrained model is not in path
            
        #train model
        print("\nTraining model...")
        for epochIndex in range(num_epoch):

            for data in train_loader:
                
                hogCollection, label = data
                
                label = label.unsqueeze(0).permute(1,0).float()
                
                hogCollection = hogCollection.float()
                
                linearOut = linearModel(hogCollection)
                
                lossVal = loss(linearOut, label)
                
                linearModelOptimizer.zero_grad()
                
                lossVal.backward()
                
                linearModelOptimizer.step()

            print('Epoch {}: Loss {}'.format(epochIndex, lossVal))

        
    predictedLabel = []
    actualLabel = []
    for data in tensorTestData:
        hogCollection, label = data

        hogCollection = hogCollection.float()
        
        linearOut = linearModel(hogCollection)
        
        # collect label predicted
        actualLabel.append(label.to("cpu").item())
        predictedLabel.append(linearOut.item())

    predictedLabelThresholded = []
    for label in predictedLabel:
        if(label >= 0.5):
            predictedLabelThresholded.append(1)
        else:  
            predictedLabelThresholded.append(0)

    #generate report
    print("Classification Report")
    print(classification_report(actualLabel, predictedLabelThresholded, target_names = ['Bad Seeds','Good Seeds']))

    print("\nConfusion Matrix")
    print(confusion_matrix(actualLabel, predictedLabelThresholded, labels=range(2)))

    actualWordLabel = convertLabel(actualLabel)
    predictedWordLabel = convertLabel(predictedLabelThresholded)

    # save as CSV File
    path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG_Linear'
    path_csv = os.path.join(path, 'classification_results.csv')

    image_paths_test = []
    for i in np.array(testData)[:, 0, 0]:
        image_paths_test.append(i)

    false_score_good_hog, false_score_bad_hog, true_score_good_hog, true_score_bad_hog, total_bad_seeds_hog, total_good_seeds_hog = save_results_csv(path, path_csv, actualWordLabel, predictedWordLabel, image_paths_test)

    print("\nTotal bad testing seeds: ", total_bad_seeds_hog)
    print("No. of Bad seeds detected correctly: ", true_score_bad_hog)
    print("No. of Bad seeds detected wrongly: ", false_score_bad_hog)

    print("\nTotal good testing seeds: ",total_good_seeds_hog)
    print("No. of Good seeds detected correctly: ", true_score_good_hog)
    print("No. of Good seeds detected wrongly: ", false_score_good_hog)
    
    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG_Linear/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG_Linear/Good_seeds/' 

    seed_set = []
    seed_type = []

    for i in np.array(image_paths_test):
        seed_set.append(i.split('/')[-3].split('S')[1]) # set of seeds
        seed_type.append(i.split('/')[-4]) # type of seeds

    save_results_corr_image(path_to_results_bad_seeds,path_to_results_good_seeds, len(testData), predictedLabelThresholded, seed_set, seed_type)


def convertLabel(labelOnes):
  outputLabel = []
  for label in labelOnes:
    if(label == 1):
      outputLabel.append("GoodSeed")
    else:
      outputLabel.append("BadSeed")
  return outputLabel



def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()

        self.l1 = torch.nn.Linear(len(trainDataHog[0]), 1)  
        # since detect wether it is good or bad, can use thresholding at 0.5 to detect good or bad seed
        

    def forward(self, x):     
        x = torch.sigmoid(self.l1(x))
        return x