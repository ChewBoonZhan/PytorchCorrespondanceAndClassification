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

#    print(testData[0][0][0])
#    print(testData[0][0][0].split('Seed')[1].split('\\')[0])

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
    print(classification_report(actualLabel, predictedLabelThresholded, target_names = ['Good Seeds','Bad Seeds']))

    actualWordLabel = convertLabel(actualLabel)
    predictedWordLabel = convertLabel(predictedLabelThresholded)

    # save as CSV File
    path = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Classification_results_HOG_Linear'
    path_csv = os.path.join(path, 'classification_results.csv')


    false_score_good=0
    false_score_bad=0
    true_score_good=0
    true_score_bad=0
    total_bad_seeds=0
    total_good_seeds=0

    if not os.path.exists(path_csv):
        os.makedirs(path)
    
    with open(path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Set","Seed","true classes", "predicted classes", "accuracy"]) # header
        
        # create test data with the good seeds
        for i in range(len(actualLabel)):
            if actualWordLabel[i] == predictedWordLabel[i]: #correct
                writer.writerow([testFilePaths[i].split('/S')[2].split('\\')[0],testFilePaths[i].split('Seed')[1].split('\\')[0], actualWordLabel[i], predictedWordLabel[i],"True"]) # get the filename in the good seed folder, and write each row with filename and 1
                if(actualWordLabel[i]=='BadSeed'):
                    true_score_bad=true_score_bad + 1
                    total_bad_seeds=total_bad_seeds + 1
                elif(actualWordLabel[i]=='GoodSeed'):
                    true_score_good=true_score_good+1
                    total_good_seeds=total_good_seeds+1
            else:
                writer.writerow([testFilePaths[i].split('/S')[2].split('\\')[0],testFilePaths[i].split('Seed')[1].split('\\')[0], actualWordLabel[i], predictedWordLabel[i],"False"])
                if(actualWordLabel[i]=='BadSeed'):
                    false_score_bad=false_score_bad+1
                    total_bad_seeds=total_bad_seeds+1
                elif(actualWordLabel[i]=='GoodSeed'):
                    false_score_good=false_score_good+1
                    total_good_seeds=total_good_seeds+1
    
    print("Total bad testing seeds: ", total_bad_seeds)
    print("No. of Bad seeds detected correctly: ", true_score_bad)
    print("No. of Bad seeds detected wrongly: ", false_score_bad)

    print("Total good testing seeds: ",total_good_seeds)
    print("No. of Good seeds detected correctly: ", true_score_good)
    print("No. of Good seeds detected wrongly: ", false_score_good)
    
    data= pd.read_csv(path_csv)

    # Save as images with red/green boxes for Classification
    path_to_badseeds = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Bad_seeds/S'
    path_to_goodseeds = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/Good_seeds/S'
    path_to_bbox_badseeds = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/Bad_seeds/S'
    path_to_bbox_goodseeds = os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/BBOX/Good_seeds/S'

    view=["top","right","left","front","rear"]

    path_to_results_bad_seeds = path + '/Bad_seeds/'
    path_to_results_good_seeds = path + '/Good_seeds/'

    isExist_bad = os.path.exists(path_to_results_bad_seeds)
    isExist_good = os.path.exists(path_to_results_good_seeds)

    if not isExist_bad:
        # Create a new directory because it does not exist 
        os.makedirs(path_to_results_bad_seeds)
        print("The new directory is created!")

    if not isExist_good:
        # Create a new directory because it does not exist 
        os.makedirs(path_to_results_good_seeds)
        print("The new directory is created!")

    good=False

    print("Saving Classification Results...")
    i=0
    # while i < 10:
    while i < len(testData):
        #set paths to img and bbox according to the set index
        if(data.iloc[i][2] == "GoodSeed"): #good seeds 9 and 10
            img_path = path_to_goodseeds + str(data.iloc[i][0]) + '/'
            bbox_path = path_to_bbox_goodseeds + str(data.iloc[i][0]) + '/'
            good = True
        else: #bad seeds 10, 11, and 12
            img_path = path_to_badseeds + str(data.iloc[i][0]) + '/'
            bbox_path = path_to_bbox_badseeds + str(data.iloc[i][0]) + '/'
            good = False
        print(bbox_path)
        for j in range(len(view)):
            print("Saving Classification Set",str(data.iloc[i][0]), str(data.iloc[i][2]) ,view[j])
            img_path_view = img_path + view[j] + '_S' + str(data.iloc[i][0]) + '.jpg'
            bbox_path_view = bbox_path +  view[j] + '/'
            x_min, y_min, x_max, y_max = readBoundingBoxCSV(bbox_path_view, True)
            numberOfSeeds = x_max.shape[0]
            img=cv2.imread(img_path_view)
            
            for index in range(numberOfSeeds): #for each seed in the seed view image
                #get its predicted label      
                pred_label = predictedLabelThresholded[i+index]
                
                #retrieve its bounding box coordinates
                x_minIndex = x_min[index]
                y_minIndex = y_min[index]
                x_maxIndex = x_max[index]
                y_maxIndex = y_max[index]
                start_point = (x_minIndex, y_minIndex)
                end_point = (x_maxIndex, y_maxIndex)

                xCenter = int(abs(x_minIndex + x_maxIndex)/2)-40
                yCenter = int(abs(y_minIndex + y_maxIndex)/2)+40

                if(pred_label==1): #predicted GoodSeed - green colour
                    color=(0,255,0)
                    text='good'
                else: #predicted BadSeed - red colour
                    color=(0,0,255)
                    text='bad'

                #draw bounding box around the seed based on its coordinates + colour according to the predicted label
                img = cv2.rectangle(img, start_point, end_point, color, 10)
                #add label (1,2,3...) for testing only
                img = cv2.putText(img, text, (xCenter,yCenter), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 10, cv2.LINE_AA)
                #after drawing bbox for each seed in the view image, save it to the directory
                name= view[j] + '_S' + str(data.iloc[i][0]) + '.jpg'
            if(good):
                cv2.imwrite(os.path.join(path_to_results_good_seeds, name),img)
            else:
                cv2.imwrite(os.path.join(path_to_results_bad_seeds, name),img)
            
        i = i + numberOfSeeds
        print("Saved.")

    


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

        print(len(trainDataHog[0]))
        self.l1 = torch.nn.Linear(len(trainDataHog[0]), 1)  
        # since detect wether it is good or bad, can use thresholding at 0.5 to detect good or bad seed
        

    def forward(self, x):     
        x = torch.sigmoid(self.l1(x))
        return x