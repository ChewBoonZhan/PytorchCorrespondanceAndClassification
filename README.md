## Computer Vision (COMP3029 UNMC) (SPM1 21-22) Coursework

### -> Cloning this repo
You can clone this repo by doing the following command:
```
git clone https://github.com/ChewBoonZhan/PytorchCorrespondanceAndClassification.git
```
<br />

### -> Pushing to this github repo
```
1. git pull origin main
2. git add .
3. git commit -m "Things to commit"
4. git push origin main
```
<br />

## Prerequisite

### -> Visual Studio Code
1. Windows: [Visual Studio](https://code.visualstudio.com/docs/setup/windows)
2. MacOS: [Visual Studio](https://code.visualstudio.com/docs/setup/mac)


### -> Downloading files
To use this project, first lets download some data and put them in the right directory, so testings can be runned on the script in this repository. 
1. Go to [here](https://drive.google.com/drive/folders/1O6xmoHd7FSzKQPwPTLaAt1ABB8oZL-iz?usp=sharing) to download 3 folders called "BBOX_Record", "Multiview_jpg", and "SIFT_Try".

2. Unzip the folders

3. Place "BBOX_Record" and "Multiview_jpg" to this directory of the cloned project:
<br /><br />
<img src = "https://i.imgur.com/GbgFU4N.png" height=300 />
<br />
As can be seen from image above, "BBOX_Record" and "Multiview_jpg" is to be extracted to the OriginalData folder.
<br /><br />

4. Place "SIFT_try" to this directory of the cloned project:
<br /><br />
<img src = "https://i.imgur.com/5asrfRH.png" height=300 />
<br />
As can be seen from image above, SIFT_try is to be extracted to ProcessedData folder.

<br /><br />

### -> Python version
The Python version required for the code to run is Python 3.7.10<br />
To install this Python version, you will first need Anaconda to make swithing between different environment easier.
You can get Anaconda [here](https://www.anaconda.com/products/distribution)
<br /><br />
To check for current Python version, type this:
```
python --version
```

To install different version of Python in Anaconda:
In the terminal, 
```
conda create --name py3 python=3.7.10
```

To activate the environment:
```
conda activate py3
```
Note: if you already have Python 3.7.10 installed, and for the subsequent running of the program, you can just run this command.
<br />

<br />

### -> Install dependencies:
#### For Windows:
1. Make sure you have Microsoft C++ Build Tools installed on your PC. You can get it [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Make sure you install the C++ Development Tools
<img src = "https://i.imgur.com/30715yI.png" height = 300/>

#### For Mac:
1. Make sure Clang is installed on your machine. For how, check this [guide](https://code.visualstudio.com/docs/cpp/config-clang-mac).

<br/>
In the project folder, there is a file called requirements.txt, which contains a list of the libraries needed to be installed. To install all the libraries:

1. Open a terminal at your project folder
2. Type:

```
1. pip install --upgrade setuptools
2. pip install -r requirements.txt
```

This shall start the installation.
<br/>
However, if you still face issue importing torch, you might need to install it manually using this command
```
1. pip uninstall torch
2. pip uninstall torchvision
3. conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

<b>Important~~</b> <br />
If during the development of the project, you face dependency issue, please search for the required dependency online, and add it to "requirements.txt" at the root of this project. After doing that, run the above command. This helps other people who dont have the dependency on their laptop to install and use the same dependency as you too when they run the command above. Thanks.


<br />


---


## Folder structures
<b>Important notice</b> <br />
Please do not leave any spaces in between folder and file names. This might interfere with running the code in Anaconda. Thanks.
<br />

1. **Preproccessing** - Script that does preliminary processing on the original seed images before correspondence and classification
2. **Classification** - Script that is used to detect classification of seeds between good and bad seed<br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; **Methods** - both satisfatory and less satisfatory methods that we've tried. <br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; The ideal one we're using is **"undefined"**. <br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; **HelperFunctions** - scipts that will be used by the methods. 

3. **Correspondance** - Scripts that was tried for estimating correspondance between 2 input seed<br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; **Methods** - both satisfatory and less satisfatory methods that we've tried. <br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; The ideal one we're using is **"Best_Exhaustive_search"**.<br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; **Bulk_processing** - scripts to run the methods for all seed sets in 1 go. <br/>
&nbsp;&nbsp;&nbsp;&nbsp;  **HelperFunctions** - scipts that will be used by the methods. 

4. **Data** - Data to used perform classification and correspondance for the coursework.<br/><br/>
&nbsp;&nbsp;&nbsp;&nbsp; **OriginalData** - original seed images and the CSV records of their bounding boxes.<br/>
&nbsp;&nbsp;&nbsp;&nbsp; **ProcessedData** - results of the preprocessing, classification, and correspondence. 

5. **General_Helper_Function** - Scripts that will be used across the program by "Preprocessing", "Classification", and "Correspondance".

<br />


---

## Running

### How to run the code?
Navigate to this project location on your machine and open a terminal. At the terminal:
```
1. Activate the python environment (Python version 3.7.10)
2. Go to "Preprocessing" folder.
3. Type "python main.py" to run the code.
4. This will crop the original seed images to remove the surroundings. 
5. Go to "Correspondance" > "Bulk_processing" > Best_Exhaustive_search"
6. Type "python main.py" to run the code. 
7. This will find the corresponding good and bad seeds for all sets from all views. 
8. This will also crop out individual seeds after correspondance between seed is detected
9. Go to "Classification" > "undefined"
10. Type "python main.py" to run the code. 
11. This will classify the seeds to be Good or Bad. A classification report should be printed out in the terminal, which states the accuracy, precision, f1-score and confusion matrix. 
```
Note: These are the steps to find seed correspondence and classification using the **ideal methods**.
To also try out our **less satisfatory methods**,

#### -> Correspondence - Seed Center + SIFT + Canny
1. Go to "Correspondance" > "Bulk_processing" > "Seed_center_line_canny_SIFT"
2. Type "python main.py" to run the code.

&nbsp;&nbsp;&nbsp;&nbsp; **To see the intermediate results**: <br/>
&nbsp;&nbsp;&nbsp;&nbsp; 1. Go to "Correspondance" > "Methods" > "Seed_center_line_canny_SIFT" <br/>
&nbsp;&nbsp;&nbsp;&nbsp; 2. Type "python main.py" to run the code.

<br/>

#### -> Correspondence - SIFT
1. Go to "Correspondance" > "Bulk_processing" > "SIFT"
2. Type "python main.py" to run the code.

&nbsp;&nbsp;&nbsp;&nbsp;**To see the intermediate results**: <br/>
&nbsp;&nbsp;&nbsp;&nbsp; 1. Go to "Correspondance" > "Methods" > "SIFT" <br/>
&nbsp;&nbsp;&nbsp;&nbsp; 2. Type "python main.py" to run the code.

<br/>

#### -> Classification - HOG + SVM
1. Go to "Classification" > "Methods" > "Hog_svm"
2. Type "python main.py" to run the code.

<br/>

#### -> Classification - Pixel + SVM
1. Go to "Classification" > "Methods" > "Pixel_svm"
2. Type "python main.py" to run the code.

<br/>

#### -> Classification - SIFT + SVM
1. Go to "Classification" > "Methods" > "SIFT_svm"
2. Type "python main.py" to run the code.

<br/>


#### -> Classification - HOG + SIFT + SVM
1. Go to "Classification" > "Methods" > "Hog_SIFT"
2. Type "python main.py" to run the code.

<br/>

#### -> Classification - HOG + DeepLearning
1. Go to "Classification" > "Methods" > "Hog_LinearLayer"
2. Type "python main.py" to run the code.

<br/>

#### -> Classification - CNN 
1. Go to "Classification" > "Methods" > "CNN"
2. Type "python main.py" to run the code.

<br/>

---

### How to get started adding code in?
You may refer [here](https://github.com/ChewBoonZhan/PytorchCorrespondanceAndClassification/blob/main/Correspondance/Methods/Example_method/main.py) to get an example of how the code works. Good luck.



