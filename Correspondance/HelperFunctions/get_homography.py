import cv2
import numpy as np

## function to compute Homography matrix based on SIFT
# Does not warp the image

def get_homography(sift_src_dict:dict, sift_dst_dict:dict):
  descriptors_src = sift_src_dict['descriptors_src']
  descriptors_dst = sift_dst_dict['descriptors_dst']
  
  keypoints_src = sift_src_dict['keypoints_src']
  keypoints_dst = sift_dst_dict['keypoints_dst']

  ###FIND MATCHING FEATURE POINTS
  MIN_MATCH_COUNT = 10 #threshold, min amount of good matched elements to have

  FLANN_INDEX_KDTREE = 1

  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #specify the algo
  search_params = dict(checks = 50) #repeat 50 times

  flann = cv2.FlannBasedMatcher(index_params, search_params) #set up the flann matcher "machine"

  # find the matches between the descriptors of each keypoints
  matches = flann.knnMatch(descriptors_src, descriptors_dst, k=2)

  good = []
  #access each element in the matches list
  for m,n in matches: #access the DM objects in each element 
    #compare the distance between the first DM obj and the second DM object
    if m.distance < 0.7*n.distance: 
        good.append(m) 

  print("Number of good matching elements: ",len(good)) 


  ####IF THERE ARE ENOUGH GOOD MATCHING POINTS BTW THE TWO VIEWS
  if len(good)>MIN_MATCH_COUNT:

    print("Computing Homography Matrix...")
    #extract the locations of good matched keypoints
    # note that keypoints are coordinats on the image, but selected for homography.
    src_pts = np.float32([ keypoints_src[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #reshape into any groups, 1 row 2 cols
    dst_pts = np.float32([ keypoints_dst[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #construct the HOMOGRAPHY matrix (projective transformation matrix) using the keypoints
    Matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # homography matrix 3x3
  else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None 


  return Matrix
