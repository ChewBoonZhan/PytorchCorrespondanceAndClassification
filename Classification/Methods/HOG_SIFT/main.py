
from combine_hog_sift import combine_hog_sift
from hog_extraction import hog_extract
from sift_extraction import sift_extract

if __name__ == '__main__':

    # called when runned from command prompt

    #extract HOG features and get bag of features for Training and Testing datasets
    train_hog_features, test_hog_features  = hog_extract()

    #extract SIFT features and get bag of features for Training and Testing datasets
    train_features, test_features, train_labels, test_labels, image_paths_test = sift_extract()

    #combine HOG bof + SIFT bof -> SVM training
    combine_hog_sift(train_hog_features, test_hog_features,train_features, test_features, train_labels, test_labels, image_paths_test)

    print("Done.")