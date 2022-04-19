import sys
import os

sys.path.insert(0, os.getcwd() + "/../HOG_SIFT/")
from sift_classification import sift_extract, evaluate_sift

if __name__ == '__main__':
    
    #classification using SIFT
    test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test = sift_extract()
    evaluate_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test)