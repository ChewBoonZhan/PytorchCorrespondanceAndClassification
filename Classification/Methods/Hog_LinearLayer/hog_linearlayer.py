import os
import sys

sys.path.insert(0, os.getcwd() + "/../../HelperFunctions/")

from dataset import loadTestData, loadTrainData

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from skimage.feature import hog
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_image_size = 128
batch_size = 8

trainDataHog = []
testDataHog = []

def hog_linearlayer():

   global trainDataHog, testDataHog

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
   #trainDataHog = []
   #numTrainData = len(trainData)
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
   #testDataHog = []
   #numTestData = len(testData)
   print("\nExtract Hog Features from Testing datasets...")
   for testDataIndex in range(len(testData)):
       testPathCollection = testData[testDataIndex]
       testDataHogIndex = []
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
   # linearModelOptimizer = torch.optim.Adam(linearModel.parameters(), lr=learning_rate)
   linearModelOptimizer = torch.optim.SGD(linearModel.parameters(), lr=0.001, momentum=0.9)
   loss = torch.nn.BCEWithLogitsLoss()

   learning_rate = 0.001
   num_epoch = 50
   
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

    
   #test model
   predictedLabel = []
   actualLabel = []
   totalData = 0
   correctData = 0
   
   print("\nTesting on trained model...")
   for data in tensorTestData:

       hogCollection, label = data

       hogCollection = hogCollection.float()

       linearOut = linearModel(hogCollection)
       
       # collect label predicted
       actualLabel.append(label.to("cpu").item())
       predictedLabel.append(linearOut.item())

       if(linearOut[0].item() >= 0.5):
           actualPredictedLabel = 1
       else:
           actualPredictedLabel = 0
           
       if(actualPredictedLabel == label):
           correctData = correctData + 1

       totalData = totalData + 1
       
   #end of testing
   accuracy = correctData/totalData * 100
   print('Accuracy: {}'.format(accuracy))

   predictedLabelThresholded = []
   for label in predictedLabel:
       if(label >= 0.5):
           predictedLabelThresholded.append(1)
       else:  
           predictedLabelThresholded.append(0)

   #generate report
   print(classification_report(actualLabel, predictedLabelThresholded, target_names = ['Bad Seeds','Good Seeds']))



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


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()

        print(len(trainDataHog[0]))
        self.l1 = nn.Linear(len(trainDataHog[0]), 2048)  
        self.l2 = nn.Linear(2048, 1)  
        # since detect wether it is good or bad, can use thresholding at 0.5 to detect good or bad seed
        

    def forward(self, x):     
        x = torch.tanh(self.l1(x))
        x = torch.sigmoid(self.l2(x))

        return x