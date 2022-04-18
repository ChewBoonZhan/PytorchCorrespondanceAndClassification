from find_correspondence import find_correspondence
import os
if __name__ == '__main__':
    #12 bad seeds 10 good seeds
    methodUsed = "lines_Canny" 
    print("Method used: ", methodUsed)

    for i in (n+1 for n in range(12)): #set number 1 to 12
        i = str(i)
        find_correspondence ("Bad_seeds", i, methodUsed)

    for i in (n+1 for n in range(10)): #set number 1 to 12
        i = str(i)
        find_correspondence ("Good_seeds", i, methodUsed)

    print("Result has been saved in:")
    print(os.getcwd() + '/../../../Data/ProcessedData/SIFT_try/_seed/S_/Results_' + methodUsed + "/")
