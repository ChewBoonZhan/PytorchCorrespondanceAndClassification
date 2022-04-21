import os
import sys

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
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
        # self.encConv2 = nn.Conv2d(4, 8, kernel_size)   # kernel size is 5
        

    def forward(self, x):

        x = F.relu(self.encConv1(x))
        # x = F.relu(self.encConv2(x))


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
        # self.l2 = nn.Linear(128, 1)  
        # since detect wether it is good or bad, can use thresholding at 0.5 to detect good or bad seed
        

    def forward(self, x):     
        x = x.view(-1, self.featureShape)
        
        x = torch.sigmoid(self.l1(x))
        # x = torch.sigmoid(self.l2(x))

        return x

linearModel = LinearClassifier().to(device)
linearModelOptimizer = torch.optim.SGD(linearModel.parameters(), lr=0.001, momentum=0.9)
loss = nn.BCELoss()



def cnn_classification():

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
    
    # change the color channel to front of image

    # if(trainDataImageIndex.shape == np.array([1]).shape):
    #   trainDataImageIndex = image
    # else:
    #   trainDataImageIndex = np.append(trainDataImageIndex, image, axis = 0)
    
            trainDataImageIndex.append(image)
        trainDataImage.append(trainDataImageIndex)
    trainDataImage = np.array(trainDataImage)


    #load Testing image into an array to be added to tensor dataset
    testDataImage = []
    for testDataIndex in range(len(testData)):
        testPathCollection = testData[testDataIndex]
        testDataImageIndex = []
        for testPath in testPathCollection:
            image = cv2.imread(testPath[0]) 
            image = torch.tensor(cv2.resize(image, (input_image_size, input_image_size))).permute(2,0,1).numpy()/255
    # change the color channel to front of image

    # if(testDataImageIndex.shape == np.array([1]).shape ):
    #   testDataImageIndex = image
    # else:
    #   testDataImageIndex = np.append(testDataImageIndex, image, axis=0)
    
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
    
    print("\nTraining model...")

    learning_rate = 0.001
    num_epoch = 50

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

    # if (random.uniform(0, 1) > 0.9) :
    #   print(featureCollection)
    #   print(linearOut)
    #   print(label)
    
            lossVal = loss(linearOut, label)
    
    # lossVal = F.cross_entropy(linearOut, label)

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
            # CNNOut = CNNOut.view(-1, featureShape)
            featureCollection = torch.cat((featureCollection, CNNOut), 0)
            
        linearOut = linearModel(featureCollection)
        # print(linearOut[0].detach().to("cpu").numpy())
        
        # collect label predicted
        actualLabel.append(label.to("cpu").item())
        predictedLabel.append(linearOut[0].detach().to("cpu").numpy()[0])
        if(linearOut[0].detach().to("cpu").numpy()[0] >= 0.5):
            actualPredictedLabel = 1
        else:
            actualPredictedLabel = 0
            
        if(actualPredictedLabel == label):
            correctData = correctData + 1
        totalData = totalData + 1

    accuracy = correctData/totalData * 100
    print('Accuracy: {}'.format(accuracy))


    predictedLabelThresholded = []
    for label in predictedLabel:
        if(label >= 0.5):
            predictedLabelThresholded.append(1)
        else:
            predictedLabelThresholded.append(0)

    #generate report
    print(classification_report(actualLabel, predictedLabelThresholded, target_names = ['Good Seeds', 'Bad Seeds']))






