"""
Author: Kilian Jakstis
Run text extraction on several demo images.
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
    main("images/img1.png")
    main("images/img2.png")
    main("images/img3.png")
    main("images/img4.png")
    main("images/img5.png")
    # main("img6.png")
    main("images/img7.png")
    main("images/img8.png")
    main("images/img9.png")
    main("images/img10.png")
    main("images/img11.png")
