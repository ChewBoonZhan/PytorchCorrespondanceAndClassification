from pixel_classification import extract_pixel_feature, evaluate_pixel

if __name__ == '__main__':

    #classification using pixel
    pixel_train, pixel_test = extract_pixel_feature()
    evaluate_pixel(pixel_train, pixel_test)
    print("Done.")