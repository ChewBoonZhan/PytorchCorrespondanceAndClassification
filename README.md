## Computer Vision (COMP3029 UNMC) (SPM1 21-22) Coursework

### Downloading files
To use this project, first lets download some data and put them in the right directory, so testings can be runned on the script in this repository. 
1. Go to [here](https://drive.google.com/drive/folders/1vkl0nrNKU9jhR6HM7tz6xdHWJxXhpII1?usp=sharing) to download image dataset used in this coursework. This dataset is different from the one in Moodle, as images have been rotated to make sure all images are in the correct orientation as the label of the image. The downloaded folder is to be inserted at "Data/OriginalData"
2. Go to [here](https://drive.google.com/drive/folders/1LVj3Y7JsPPjD08F3M05E4tFVSDKpAD-g?usp=sharing) to download Bounding Box that allows 100% accuracy for seed detection. This is crucial as bounding box will be used as a vital tool in correspondance estimation between 2 given images. The downloaded folder is to be inserted at "Data/OriginalData"
3. Go to [here](https://drive.google.com/drive/folders/1sL1AARlqwZPi_I6JEL7LvuVeyqBMamsq?usp=sharing) to download image dataset, and bounding box for image with side noise cropped out. This allows side noise of the image to be cropped away. The downloaded folder is to be inserted at "Data/ProcessedData"
<br /><br />

### To install dependencies:
```
pip install -r requirements.txt
```
<b>Important~~</b> <br />
If during the development of the project you face dependency issue, please search for the required dependency online, and add it to "requirements.txt" at the root of this project. After doing that, run the above command. This helps other people who dont have the dependecy on their laptop to install and use the same dependency as you too when they run the command above. Thanks.


<br />

### Folder structures
1. Classification - Script that was tried for classification of seeds between good and bad seed
2. Correspondance - Scripts that was tried for estimating correspondance between 2 input seed
3. Data - Data to used perform classification and correspondance for the coursework.

### How to get started adding code in?
You may refer [here](https://github.com/ChewBoonZhan/PytorchCorrespondanceAndClassification/blob/main/Correspondance/Methods/Best_Exhaustive_search/main.py) to get an example of how the code works. Good luck.
