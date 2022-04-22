## Classification using Pixel & SVM

In this method, we are extracting the pixel value as feature for seed images. Each seed is a set of correspondance, such that we have the left, right, top, rear and front seed feature extracted using HOG, and its features are concatenated before passed through SVM.

<br /><br />
To run the code:
```
python main.py
```