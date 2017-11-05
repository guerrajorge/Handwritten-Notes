import cv2

## WORKS
img_name = 'datasets/GO1_morph_2.png'
# img = cv2.imread(img_name)
# mser = cv2.MSER_create()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = img.copy()
#
# regions = mser.detectRegions(gray, None)
# print('len of regions = {0}'.format(len(regions)))
# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[760:794]]
# cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
# cv2.imwrite('datasets/GO1_morph_2_detect_2.png', gray_img)

### WORKS WELL
# import numpy as np
#
# mser = cv2.MSER_create()
# img = cv2.imread(img_name)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# vis = img.copy()
# regions = mser.detectRegions(gray, None)
#
# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
#
# cv2.imshow('img', vis)
#
# mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
#
# for contour in hulls:
#
#     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
#
# #this is used to find only text regions, remaining are ignored
# text_only = cv2.bitwise_and(img, img, mask=mask)
#
# cv2.imshow("text only", text_only)
#
# cv2.imwrite('datasets/textonly.png', text_only)



import cv2


def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imshow('datasets/imageresults.png', img)
    cv2.waitKey()


captch_ex(img_name)
