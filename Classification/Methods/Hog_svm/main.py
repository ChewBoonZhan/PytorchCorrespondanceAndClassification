from hog_classification import extract_hog_feature, evaluate_hog

if __name__ == '__main__':

    #classification using HOG
    hog_train, hog_test = extract_hog_feature()
    evaluate_hog(hog_train, hog_test)
    print("Done.")