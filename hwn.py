import argparse
import logging
from utils.logger import logger_initialization
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc.pilutil import imresize
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


def split_2d(img, cell_size, flatten=True):
    height, width = img.shape[:2]
    size_x, size_y = cell_size
    cells = [np.hsplit(row, width // size_x) for row in np.vsplit(img, height // size_y)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, size_y, size_x)
    return cells


def load_digits(fn):
    # size of each digit is SZ x SZ
    digit_dimension = 20
    # 0-9
    number_class = 10

    print('loading "{0} for training" ...'.format(fn))
    digits_img = cv2.imread(fn, 0)
    digits = split_2d(digits_img, (digit_dimension, digit_dimension))
    labels = np.repeat(np.arange(number_class), len(digits) / number_class)
    # 2500 samples in the digits.png so repeat 0-9 2500/10(0-9 - no. of classes) times.
    return digits, labels


def pixels_to_hog_20(pixel_array):
    hog_features_data = list()
    for img in pixel_array:
        # img = 20x20
        fd = hog(img, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualise=False,
                 block_norm='L1')
        hog_features_data.append(fd)
    hog_features = np.array(hog_features_data, 'float64')
    return np.float32(hog_features)


def contains(r1, r2):
    r1_x1 = r1[0]
    r1_y1 = r1[1]
    r2_x1 = r2[0]
    r2_y1 = r2[1]

    r1_x2 = r1[0] + r1[2]
    r1_y2 = r1[1] + r1[3]
    r2_x2 = r2[0] + r2[2]
    r2_y2 = r2[1] + r2[3]

    # does r1 contain r2?
    return r1_x1 < r2_x1 < r2_x2 < r1_x2 and r1_y1 < r2_y1 < r2_y2 < r1_y2


def get_digits(contours):
    digit_rects = [cv2.boundingRect(ctr) for ctr in contours]
    rects_final = digit_rects[:]

    for r in digit_rects:
        x, y, w, h = r
        if w < 15 and h < 15:  # too small, remove it
            rects_final.remove(r)

    for r1 in digit_rects:
        for r2 in digit_rects:
            if (r1[1] != 1 and r1[1] != 1) and (r2[1] != 1 and r2[1] != 1):
                # if the rectangle is not the page-bounding rectangle,
                if contains(r1, r2) and (r2 in rects_final):
                    rects_final.remove(r2)
    return rects_final


def process_test_image(dataset, model, model_type='dnn'):
    logging.getLogger('regular.time').info('loading "{0} for digit recognition" ...'.format(dataset))
    im = cv2.imread(dataset)
    im_original = cv2.imread(dataset)

    blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    digits_rect = get_digits(contours)  # rectangles of bounding the digits in user image

    for rect in digits_rect:
        x, y, w, h = rect
        _ = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        im_digit = im_original[y: y + h, x: x + w]
        sz = 28
        im_digit = imresize(im_digit, (sz, sz))

        for i in range(sz):  # need to remove border pixels
            im_digit[i, 0] = 255
            im_digit[i, 1] = 255
            im_digit[0, i] = 255
            im_digit[1, i] = 255

        thresh = 210
        im_digit = cv2.cvtColor(im_digit, cv2.COLOR_BGR2GRAY)
        im_digit = cv2.threshold(im_digit, thresh, 255, cv2.THRESH_BINARY)[1]
        # im_digit = cv2.adaptiveThreshold(im_digit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,2)
        im_digit = (255 - im_digit)

        im_digit = imresize(im_digit, (20, 20))

        hog_img_data = pixels_to_hog_20([im_digit])

        pred = model.predict(hog_img_data)

        if pred.shape[1] == 10:
            pred = pred.ravel()

        _ = cv2.putText(im, str(pd.Series(pred).idxmax()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        _ = cv2.putText(blank_image, str(int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    cv2.imwrite("original_overlay.png", im)
    cv2.imwrite("final_digits.png", blank_image)
    cv2.destroyAllWindows()


def main():
    # get the the path for the input file argument
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-tr', '--train_image', help='image used for training the algorithm', required=True)
    required.add_argument('-te', '--test_image', help='image to evaluate', required=True)
    optional.add_argument('-l', '--log', dest="logLevel", choices=['DEBUG', 'debug', 'INFO', 'info', 'ERROR', 'error'],
                          help='Argument use to set the logging level')
    optional.add_argument('-knn', '--knn', help='flag to run knn', action='store_true')

    args = parser.parse_args()

    logger_initialization(log_level=args.logLevel)

    logging.getLogger('regular.time').info('starting running handwritten-notes script')

    digits, y_train = load_digits(args.train_image)

    x_train = pixels_to_hog_20(digits)

    num_pixels = x_train.shape[1]
    num_classes = len(np.unique(y_train))

    if args.knn:
        logging.getLogger('regular.time').info('training knn model')
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)
    else:
        logging.getLogger('regular.time').info('training NN model')
        model = Sequential()
        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))   
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    process_test_image(dataset=args.test_image, model=model, model_type=args.knn)


if __name__ == '__main__':
    main()
