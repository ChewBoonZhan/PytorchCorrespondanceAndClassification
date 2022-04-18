import os
import csv

def loadTrainData():
    os.chdir('C:/Users/HuiFang/Desktop/Year 3/Spring/Computer Vision/CW/PytorchCorrespondanceAndClassification/Data/ProcessedData')
    #Path Example: SIFT_try/Training/Good_seeds/S1/Seed1/top.jpg
    training_path_bad = 'SIFT_try/Training/Bad_seeds'
    training_path_good = 'SIFT_try/Training/Good_seeds'

    #a list of img paths + labels where each element in the list is the paths to all views for 1 seed 
    trainData=[]
    #Example:
    #[[(seed1 top S1,0),(seed1 right S1,0),...],
    # [(seed2 top S1, 0),(seed2 right S1,0),...], 
    # [..S2..], [..S3..]]

    img_label_list=[] #a list of labels 0/1. 1 seed 1 label

    # with open('SIFT_try/Training/trainingdata.csv', 'w', newline='') as file:
    with open('training_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"]) # header
    
        for i in (n+1 for n in range(9)): #Set 1 to 9 for Bad seeds
    
            print('Loading Training Bad seeds Set' + str(i))

            path_to_set = training_path_bad + '/S' + str(i) # 'SIFT_try/Training/Bad_seeds/S1'

            for seed in os.listdir(path_to_set): #return Seed1, Seed2, Seed3, ...

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Training/Bad_seeds/S1/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in os.listdir(path_to_seed): #return top.jpg, right.jpg,...

                    path_to_seed_view = os.path.join(path_to_seed, view) #'.../S1/Seed1/top.jpg'
        
                    #read path to get image
                    #image = cv2.imread(path_to_seed_view)
                    #img_seed.append((image,0))

                    #if want paths instead of image (arrays), use the following
                    img_seed.append((path_to_seed_view,0)) #['.../S1/Seed1/top.jpg', '.../S1/Seed1/right.jpg',....]
                    #filename = 'S'+str(i)+'_'+seed+view #S1Seed1top.jpg
                    writer.writerow([path_to_seed_view, 0])

                trainData.append(img_seed)
        #img_seed_list.append(img_seed)
        #img_label_list.append(0)

        for i in (n+1 for n in range(8)): #Set 1 to 8 for Good seeds
        
            print('Loading Training Good seeds Set' + str(i))

            path_to_set = training_path_good + '/S' + str(i) # 'SIFT_try/Training/Good_seeds/S1'

            for seed in os.listdir(path_to_set): #return Seed1, Seed2, Seed3, ...

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Training/Good_seeds/S1/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in os.listdir(path_to_seed): #return top.jpg, right.jpg,...

                    path_to_seed_view = os.path.join(path_to_seed, view) #'.../S1/Seed1/top.jpg'
            
                    #read path to get image
                    #image = cv2.imread(path_to_seed_view)
                    #img_seed.append((image,1))

                    #if want paths instead of image (arrays), use the following
                    img_seed.append((path_to_seed_view,1)) #['.../S1/Seed1/top.jpg', '.../S1/Seed1/right.jpg',....]
                    #filename = 'S'+str(i)+'_'+seed+view #S1Seed1top.jpg
                    writer.writerow([path_to_seed_view, 1])

                trainData.append(img_seed)
                #img_seed_list.append(img_seed)
                #img_label_list.append(1)

    print('Training Dataset created.')
    return trainData

def loadTestData():
    os.chdir('C:/Users/HuiFang/Desktop/Year 3/Spring/Computer Vision/CW/PytorchCorrespondanceAndClassification/Data/ProcessedData')
    #Path Example: SIFT_try/Testing/Good_seeds/S9/Seed1/top.jpg
    testing_path_bad = 'SIFT_try/Testing/Bad_seeds'
    testing_path_good = 'SIFT_try/Testing/Good_seeds'

    views=['top','right','left','front','rear'] #load testing seeds in this exact order

    testData=[]
    img_label_list=[] 

    with open('SIFT_try/Testing/testingdata.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"]) # header

        for i in (n+10 for n in range(3)): #Set 10 to 12 for Bad seeds
        
            print('Extract Testing Bad seeds Set' + str(i))

            path_to_set = testing_path_bad + '/S' + str(i) # 'SIFT_try/Testing/Bad_seeds/S10'

            for seed in os.listdir(path_to_set): #return Seed1, Seed2, Seed3, ...

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Testing/Bad_seeds/S10/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...

                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S10/Seed1/top.jpg'
        
                    #get image from path
                    #image = cv2.imread(path_to_seed_view)
                    #img_seed.append((image,0))

                    img_seed.append((path_to_seed_view,0)) #['.../S10/Seed1/top.jpg', '.../S10/Seed1/right.jpg',....]
                    #filename = 'S'+str(i)+'_'+seed+view #S1Seed1top.jpg
                    writer.writerow([path_to_seed_view, 0])

                testData.append(img_seed)
                #img_label_list.append(0)

        for i in (n+9 for n in range(2)): #Set 9 to 10 for Good seeds
        
            print('Extract Testing Good seeds Set' + str(i))

            path_to_set = testing_path_good + '/S' + str(i) # 'SIFT_try/Testing/Good_seeds/S9'

            for seed in os.listdir(path_to_set): #return Seed1, Seed2, Seed3, ...

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Testing/Good_seeds/S9/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...

                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S9/Seed1/top.jpg'
        
                    #get image from path
                    #image = cv2.imread(path_to_seed_view)
                    #img_seed.append((image,1))

                    img_seed.append((path_to_seed_view,1)) #['.../S9/Seed1/top.jpg', '.../S9/Seed1/right.jpg',....]
                    #filename = 'S'+str(i)+'_'+seed+view #S1Seed1top.jpg
                    writer.writerow([path_to_seed_view, 1])

                testData.append(img_seed)
                #img_label_list.append(1)

    print('Testing dataset created')
    return testData

if __name__ == '__main__':
    os.chdir('C:/Users/HuiFang/Desktop/Year 3/Spring/Computer Vision/CW/PytorchCorrespondanceAndClassification/Data/ProcessedData')
    # called when runned from command prompt
    trainData = loadTrainData()
    testData = loadTestData()