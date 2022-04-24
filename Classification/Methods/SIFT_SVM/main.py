import sys
import os

from sift_classification import sift_extract, evaluate_sift

if __name__ == '__main__':
    
    #classification using SIFT
    sift_extract()
    print("Done.")