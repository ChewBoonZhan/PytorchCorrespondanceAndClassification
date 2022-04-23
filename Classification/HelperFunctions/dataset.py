import os
import csv

def loadTrainData():

    #Path Example: SIFT_try/Training/Good_seeds/S1/Seed1/top.jpg
    training_path_bad = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Training/Bad_seeds"
    training_path_good = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Training/Good_seeds"

    views=['top','right','left','front','rear'] #load training seeds in this exact order

    #a list of img paths + labels where each element in the list is the paths to all views for 1 seed 
    trainData=[]

    with open(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Training/trainingdata.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"]) # header
    
        for i in (n+1 for n in range(9)): #Set 1 to 9 for Bad seeds
    
            print('\nLoading Training Bad seeds Set' + str(i))

            path_to_set = training_path_bad + '/S' + str(i) # 'SIFT_try/Training/Bad_seeds/S1'
            
            numberOfSeeds=len([name for name in os.listdir(path_to_set)])
  
            for seed_index in (n+1 for n in range(numberOfSeeds)):
                
                seed = "Seed" + str(seed_index)

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Training/Bad_seeds/S1/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...
                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S1/Seed1/top.jpg'
                    path_to_seed_view = path_to_seed_view.replace(os.sep, "/") # use same os separator for filename in Windows and Mac
                    img_seed.append((path_to_seed_view,0)) #['.../S1/Seed1/top.jpg', '.../S1/Seed1/right.jpg',....]
                    writer.writerow([path_to_seed_view, 0])

                trainData.append(img_seed)

        for i in (n+1 for n in range(8)): #Set 1 to 8 for Good seeds
        
            print('\nLoading Training Good seeds Set' + str(i))

            path_to_set = training_path_good + '/S' + str(i) # 'SIFT_try/Training/Good_seeds/S1'

            numberOfSeeds=len([name for name in os.listdir(path_to_set)])
  
            for seed_index in (n+1 for n in range(numberOfSeeds)):
                
                seed = "Seed" + str(seed_index)

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Training/Good_seeds/S1/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...
                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S1/Seed1/top.jpg'
                    path_to_seed_view = path_to_seed_view.replace(os.sep, "/") # use same os separator for filename in Windows and Mac
                    img_seed.append((path_to_seed_view,1)) #['.../S1/Seed1/top.jpg', '.../S1/Seed1/right.jpg',....]
                    writer.writerow([path_to_seed_view, 1])

                trainData.append(img_seed)

    print('\nTraining Dataset created.')
    return trainData

def loadTestData():

    #Path Example: SIFT_try/Testing/Good_seeds/S9/Seed1/top.jpg
    testing_path_bad = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Testing/Bad_seeds"
    testing_path_good = os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Testing/Good_seeds"

    views=['top','right','left','front','rear'] #load testing seeds in this exact order

    testData=[]

    with open(os.getcwd() + "/../../../Data/ProcessedData/SIFT_try/Testing/testingdata.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "label"]) # header

        for i in (n+10 for n in range(3)): #Set 10 to 12 for Bad seeds
        
            print('\nExtract Testing Bad seeds Set' + str(i))

            path_to_set = testing_path_bad + '/S' + str(i) # 'SIFT_try/Testing/Bad_seeds/S10'

            numberOfSeeds=len([name for name in os.listdir(path_to_set)])
  
            for seed_index in (n+1 for n in range(numberOfSeeds)):

                seed = "Seed" + str(seed_index)

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Testing/Bad_seeds/S10/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...
                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S10/Seed1/top.jpg'
                    path_to_seed_view = path_to_seed_view.replace(os.sep, "/") # use same os separator for filename in Windows and Mac
                    img_seed.append((path_to_seed_view,0)) #['.../S10/Seed1/top.jpg', '.../S10/Seed1/right.jpg',....]
                    writer.writerow([path_to_seed_view, 0])

                testData.append(img_seed)

        for i in (n+9 for n in range(2)): #Set 9 to 10 for Good seeds
        
            print('\nExtract Testing Good seeds Set' + str(i))

            path_to_set = testing_path_good + '/S' + str(i) # 'SIFT_try/Testing/Good_seeds/S9'

            numberOfSeeds=len([name for name in os.listdir(path_to_set)])
  
            for seed_index in (n+1 for n in range(numberOfSeeds)):
                
                seed = "Seed" + str(seed_index)

                path_to_seed = os.path.join(path_to_set,seed) # 'SIFT_try/Testing/Good_seeds/S9/Seed1'

                img_seed=[] #paths for all views per seed 

                for view in views: #return top.jpg, right.jpg,...

                    view_name= view + '.jpg'
                    path_to_seed_view = os.path.join(path_to_seed, view_name) #'.../S9/Seed1/top.jpg'
                    path_to_seed_view = path_to_seed_view.replace(os.sep, "/") # use same os separator for filename in Windows and Mac
                    img_seed.append((path_to_seed_view,1)) #['.../S9/Seed1/top.jpg', '.../S9/Seed1/right.jpg',....]
                    writer.writerow([path_to_seed_view, 1])

                testData.append(img_seed)

    print('\nTesting dataset created.')
    return testData