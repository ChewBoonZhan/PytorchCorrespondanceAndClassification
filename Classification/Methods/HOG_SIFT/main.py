
from combine_hog_sift import combine_hog_sift
from hog import hog, evaluate_hog
from sift import sift, evaluate_sift

if __name__ == '__main__':

    # called when runned from command prompt

    #classification using HOG
    y_test, y_pred, true_classes_hog, predict_classes_hog, image_path_test_hog = hog()
    evaluate_hog(y_test, y_pred, true_classes_hog, predict_classes_hog, image_path_test_hog)

    #classification using SIFT
    test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test = sift()
    evaluate_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test)

    #combined HOG+SIFT
    combine_hog_sift(test_labels, pred_labels, true_classes_sift, predict_classes_sift, image_paths_test, y_pred,predict_classes_hog)

    print("Done")