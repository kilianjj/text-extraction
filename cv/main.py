"""
Test on several images.
"""

import util
import cv2 as cv

def main(img_path):
    # load image
    img = cv.imread(img_path)
    # get characters
    chars = util.get_characters(img)
    # get bounding boxes
    util.bounding_boxes(img, chars)
    # get words
    words = util.find_words(chars)
    text = ' '.join([word for word in words])
    cleaned_text = util.clean_text(text)
    print(cleaned_text)
    # save the image
    im_name = img_path.split('.')[0]
    cv.imwrite(f'{im_name}_output.png', img)

if __name__ == '__main__':
    main("img1.png")
    main("img2.png")
    main("img3.png")
    main("img4.png")
    main("img5.png")
    # main("img6.png")
    main("img7.png")
    main("img8.png")
    main("img9.png")
    main("img10.png")
