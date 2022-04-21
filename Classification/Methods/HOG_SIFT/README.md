## Classification using HOG & SIFT & SVM

In this method, we are using both HOG and SIFT as feature extractor for seed images. 
<br/>
Bag of features are created using HOG and SIFT features respectively. Then, the HoG Bag of Features and SIFT Bag of Features of each seed are concatenated to form a combined HOG_SIFT bag of features. 
<br/>
Such bag of features from the Training seeds are then passed through a Linear SVC model.

<br /><br />
To run the code:
```
python main.py
```