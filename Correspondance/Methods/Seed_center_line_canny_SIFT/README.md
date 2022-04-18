## Generate line between seed center and find homography using SIFT
This folder contains function to allow user to find homography matrix using SIFT. Both source image and destination image first have their seed center used to generate lines between them. This allows us to form a structure which shows the distribution of seed in the image. The line image is then used to detect canny, which the canny edge is added back to the image.
<br /><br />
To run the code:
```
python main.py
```