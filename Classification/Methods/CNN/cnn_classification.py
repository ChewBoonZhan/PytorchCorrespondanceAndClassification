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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from save_results_csv import save_results_csv
from save_results_corr_image import save_results_corr_image

import csv
import pandas as pd

sys.path.insert(0, os.getcwd() + "/../../../General_Helper_Function/")

from readBoundingBoxCSV import readBoundingBoxCSV


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Training parameters
batch_size = 16
input_image_size = 128

#CNN params
imgChannels = 3
kernel_size = 3

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


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, kernel_size)   # kernel size is 5

    def forward(self, x):

        x = F.relu(self.encConv1(x))


        return x


CNNModel = CNNClassifier().to(device)
CNNModelOptimizer = torch.optim.SGD(CNNModel.parameters(), lr=0.001, momentum=0.9)

class LinearClassifier(nn.Module):

    def __init__(self):
        super(LinearClassifier, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
       
        dummydata = torch.rand(1, imgChannels, input_image_size, input_image_size).to(device)
        outFeature = CNNModel(dummydata)

        self.featureShape = outFeature.shape[0] * outFeature.shape[1] * outFeature.shape[2] * outFeature.shape[3]  # cause we have 5
        
        self.l1 = nn.Linear(self.featureShape, 1)  
        # since detect wether it is good or bad, can use thresholding at 0.5 to detect good or bad seed
        

    def forward(self, x):     
        x = x.view(-1, self.featureShape)
        
        x = torch.sigmoid(self.l1(x))

        return x

linearModel = LinearClassifier().to(device)
linearModelOptimizer = torch.optim.SGD(linearModel.parameters(), lr=0.001, momentum=0.9)
loss = nn.BCELoss()

testFilePaths= []

def cnn_classification():
    global testFilePaths, testData

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

    #load Training image into an array to be added to tensor dataset
    trainDataImage = []
    for trainDataIndex in range(len(trainData)):  
        trainPathCollection = trainData[trainDataIndex]
        trainDataImageIndex = []
        for trainPath in trainPathCollection:
            image = cv2.imread(trainPath[0]) 
            image = torch.tensor(cv2.resize(image, (input_image_size, input_image_size))).permute(2,0,1).numpy()/255
    
            trainDataImageIndex.append(image)
        trainDataImage.append(trainDataImageIndex)
    trainDataImage = np.array(trainDataImage)

    
    #load Testing image into an array to be added to tensor dataset
    testDataImage = []
    for testDataIndex in range(len(testData)):
        testPathCollection = testData[testDataIndex]
        testDataImageIndex = []
        testFilePaths.append(testPathCollection[0][0])
        for testPath in testPathCollection:
            image = cv2.imread(testPath[0]) 
            image = torch.tensor(cv2.resize(image, (input_image_size, input_image_size))).permute(2,0,1).numpy()/255
            # change the color channel to front of image
    
            testDataImageIndex.append(image)
        testDataImage.append(testDataImageIndex)
    testDataImage = np.array(testDataImage)


    train_loader = tensor_train(trainDataImage,trainLabel)
    tensorTestData = tensor_test(testDataImage,testLabel)

    run_CNN(train_loader, tensorTestData)


def tensor_train(trainDataImage,trainLabel):

    trainDataImage = torch.tensor(np.array(trainDataImage))
    trainLabel = torch.tensor(trainLabel)
    # creating training tensor dataset
    tensorTrainData = TensorDataset(trainDataImage, trainLabel)
    # creating data loader for data
    train_loader = DataLoader(tensorTrainData, batch_size, shuffle=True)
    # send data loaded to device
    train_loader = DeviceDataLoader(train_loader, device)

    return train_loader


def tensor_test(testDataImage,testLabel):

    testDataImage = torch.tensor(testDataImage)
    testLabel = torch.tensor(testLabel)
    # creating testing tensor dataset
    tensorTestData = TensorDataset(testDataImage, testLabel)
    tensorTestData = DeviceDataLoader(tensorTestData, device)

    return tensorTestData
    

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def run_CNN(train_loader, tensorTestData):
    
    

    learning_rate = 0.001
    num_epoch = 50
    if(os.path.exists("CNN.pth") and os.path.exists("linear.pth")):
        print("\Loading existing model...")
        # load the model
        CNNModel.load_state_dict(torch.load("CNN.pth", map_location=device))
        linearModel.load_state_dict(torch.load("linear.pth", map_location=device))
    else:
        print("\nTraining model...")
        for epochIndex in range(num_epoch):
            for data in train_loader:

                imgCollection, label = data
                imgCollection = imgCollection.float()
                label = label.unsqueeze(0).permute(1,0).float()    
                imgCollection = imgCollection.permute(1, 0, 2, 3, 4).float()
                
                featureCollection = torch.tensor([]).to(device)
                for image in imgCollection:
                    # train classifier
                    CNNOut = CNNModel(image).unsqueeze(0)
                    # CNNOut = CNNOut.view(-1, featureShape)
                    featureCollection = torch.cat((featureCollection, CNNOut), 0)
                    
                featureCollection = torch.mean(featureCollection, dim = 0)

                linearOut = linearModel(featureCollection)
        
                lossVal = loss(linearOut, label)

                CNNModelOptimizer.zero_grad()
                linearModelOptimizer.zero_grad()

                lossVal.backward()

                CNNModelOptimizer.step()
                linearModelOptimizer.step()

    
            print('Epoch {}: Loss {}'.format(epochIndex, lossVal))

        
    print("\nTesting on trained model...")

    predictedLabel = []
    actualLabel = []
    totalData = 0
    correctData = 0
    
    for data in tensorTestData:
        imgCollection, label = data
        imgCollection = imgCollection.unsqueeze(0).permute(1, 0, 2, 3, 4).float()
        featureCollection = torch.tensor([]).to(device)
        
        for image in imgCollection:
            # train classifier
            CNNOut = CNNModel(image)
            featureCollection = torch.cat((featureCollection, CNNOut), 0)
            
        linearOut = linearModel(featureCollection)
        
        # collect label predicted
        actualLabel.append(label.to("cpu").item())
        predictedLabel.append(linearOut[0].detach().to("cpu").numpy()[0])


    predictedLabelThresholded = []
    for label in predictedLabel:
        if(label >= 0.5):
            predictedLabelThresholded.append(1)
        else:
            predictedLabelThresholded.append(0)

    #generate report
    print(classification_report(actualLabel, predictedLabelThresholded, target_names = ['Bad Seeds', 'Good Seeds']))
    print("\nConfusion Matrix")
    print(confusion_matrix(actualLabel, predictedLabelThresholded, labels=range(2)))
    
    
    actualWordLabel = convertLabel(actualLabel)
    predictedWordLabel = convertLabel(predictedLabelThresholded)

    path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Classification_results_CNN'
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
    
    path_to_results_bad_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_CNN/Bad_seeds/'
    path_to_results_good_seeds = os.getcwd()+ '/../../../Data/ProcessedData/SIFT_try/Classification_results_CNN/Good_seeds/' 

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





