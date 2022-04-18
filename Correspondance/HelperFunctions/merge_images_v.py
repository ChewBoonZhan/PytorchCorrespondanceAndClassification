import cv2
import numpy as np

def merge_images_v(img1, img2):
  # load images
  w1 = img1.shape[1]
  w2 = img2.shape[1]

  # get maximum width
  ww = max(w1, w2)

  # pad images with transparency in width
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
  img1 = cv2.copyMakeBorder(img1, 0, 0, 0, ww-w1, borderType=cv2.BORDER_CONSTANT, value=(0,0,0,0))
  img2 = cv2.copyMakeBorder(img2, 0, 0, 0, ww-w2, borderType=cv2.BORDER_CONSTANT, value=(0,0,0,0))

  # stack images vertically
  result = cv2.vconcat([img1, img2])

  return result