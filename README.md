## Computer Vision (COMP3029 UNMC) (SPM1 21-22) Coursework

### -> Cloning this repo
You can clone this repo by doing the following command:
```
git clone https://github.com/ChewBoonZhan/PytorchCorrespondanceAndClassification.git
```
<br />

### -> Pushing to the github repo
```
1. git pull origin main
2. git add .
3. git commit -m "Things to commit"
4. git push origin main
```
<br />

### -> Downloading files
To use this project, first lets download some data and put them in the right directory, so testings can be runned on the script in this repository. 
1. Go to [here](https://drive.google.com/drive/folders/1O6xmoHd7FSzKQPwPTLaAt1ABB8oZL-iz?usp=sharing) to download 2 folders called "BBOX_Record" and "Multiview_jpg". 

2. Extract the data here, in this format:
<br /><br />
<img src = "Data/OriginalData/Image/1.png" height=300 />
<br />
As can be seen from image above, "BBOX_Record" and "Multiview_jpg" is to be extracted to OriginalData folder.

3. Go to [here](https://drive.google.com/drive/folders/1O6xmoHd7FSzKQPwPTLaAt1ABB8oZL-iz?usp=sharing) to download a folder called "SIFT_Try". 

4. Extract the data here, in this format:
<br /><br />
<img src = "Data/ProcessedData/Image/1.png" height=300 />
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
```
1. conda create --name py3 python=3.7.10
2. conda activate py3
```
<br />

Therefore, if you already have Python 3.7.10 installed, and wish to just change to its environment, you can just run the following command:

```
conda activate py3
```
<br />

### -> To install dependencies:
```
pip install -r requirements.txt
```
<b>Important~~</b> <br />
If during the development of the project you face dependency issue, please search for the required dependency online, and add it to "requirements.txt" at the root of this project. After doing that, run the above command. This helps other people who dont have the dependecy on their laptop to install and use the same dependency as you too when they run the command above. Thanks.


<br />

### -> Folder structures
<b>Important notice</b> <br />
Please do not leave any spaces in between folder and file names. This might interfere with running the code in Anaconda. Thanks.
<br />

1. Classification - Script that was tried for classification of seeds between good and bad seed
2. Correspondance - Scripts that was tried for estimating correspondance between 2 input seed
3. Data - Data to used perform classification and correspondance for the coursework.

<br />

### -> How to get started adding code in?
You may refer [here](https://github.com/ChewBoonZhan/PytorchCorrespondanceAndClassification/blob/main/Correspondance/Methods/Example_method/main.py) to get an example of how the code works. Good luck.
