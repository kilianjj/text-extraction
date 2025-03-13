"""
Utility functions for processing images and characters.
Includes functions for extracting characters, classifying characters, and finding words in an image.
"""

import cv2 as cv
import numpy as np
from model import Classifier
import config

class Character:
    """
    Class for representing a character.
    """

    empty = ""
    median_width = 0
    median_height = 0

    def __init__(self, centroid, x, y, w, h, scaled_img, classification=empty):
        """
        Initialize a character with a location, size, image, and classification that empty by default.
        :param centroid: centroid of the character in original image
        :param x: min x coordinate of the character in original image
        :param y: min y coordinate of the character in original image
        :param w: width in original image
        :param h:  height in original image
        :param scaled_img: the normalized image of the character
        :param classification: predicted classification of the character
        """
        self.centroid = centroid
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.scaled_img = scaled_img
        self.classification = classification

    def classify(self, model):
        """
        Classify the character using the given model.
        """
        num_class = model.predict(self.scaled_img)
        self.classification = self.map_classification(num_class)

    @staticmethod
    def map_classification(val):
        # convert by-class index to character
        emnist__mapping = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
            36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
            46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
            56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
        }
        return emnist__mapping.get(val, '')

    @staticmethod
    def scale_char_image(img):
        """
        Resize the character image to be optimal for classification.
        """

        # method one:
        # pad the image to make it square while keeping original aspect ratio
        diff = abs(img.shape[0] - img.shape[1])
        if img.shape[0] > img.shape[1]:  # Taller than wide
            pad_left = diff // 2
            pad_right = diff - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant')
        elif img.shape[1] > img.shape[0]:  # Wider than tall
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant')

        # scale and center
        w, h = config.model_input_size
        result = np.zeros(config.model_input_size, dtype=np.uint8)
        resized = cv.resize(img, (w - 4, h - 4))
        result[2:h - 2, 2:w - 2] = resized

        # rotate and flip to match training data orientation
        result = cv.rotate(result, cv.ROTATE_90_CLOCKWISE)
        result = cv.flip(result, 1)

        # method two:
        # w, h = config.model_input_size
        # result = np.zeros(config.model_input_size, dtype=np.uint8)
        # resized = cv.resize(img, (w - 4, h - 4))
        # result[2:h - 2, 2:w - 2] = resized

        return result

    def __str__(self):
        return f'Character {self.classification} at {self.centroid}'

    def is_contiguous(self, other):
        """
        Check if the character is next to the previous one (ordered from top left to bottom right).
        """
        first_end = other.x + other.width
        second_start = self.x
        # see if the x distance between them is less than half the median width of chapters
        return abs(first_end - second_start) < config.median_width_scalar * Character.median_width

def clean_text(text):
    """
    Try to get rid of simple mistakes like having a 1 instead of an 'I' in a word with simple rules.
    """
    new = ''
    for i in range(len(text)):
        if i == len(text) - 1:
            if text[i] == '1':
                new += 'I' if text[i - 1].isalpha() else '1'
                continue
            if text[i] == '0':
                new += 'O' if text[i - 1].isalpha() else '0'
                continue
            new += text[i]
            continue
        if i == 0:
            if text[i] == '1':
                new += 'I' if text[i + 1].isalpha() else '1'
                continue
            if text[i] == '0':
                new += 'O' if text[i + 1].isalpha() else '0'
                continue
            new += text[i]
            continue

        if text[i] == '1' or text[i] == '0':
            if i == 0:
                new += 'I' if text[i + 1].isalpha() else 'O'
            if text[i - 1].isalpha() or text[i + 1].isalpha():
                new += 'I' if text[i] == '1' else 'O'
        else:
            new += text[i]
    return new

def extract_words(line):
    """
    Get letters that are horizontally close and in the same line to form words.
    """
    words = []
    current_word = []
    x_sorted = sorted(line, key=lambda x: x.centroid[0])
    for i, char in enumerate(x_sorted):
        if i == 0:
            current_word.append(char)
            continue
        if char.is_contiguous(x_sorted[i - 1]):
            current_word.append(char)
        else:
            words.append(current_word)
            current_word = [char]
    words.append(current_word)
    return words

def extract_text(characters):
    """
    Identify lines by grouping characters that are vertically close.
    Then do a similar process to group characters that are horizontally close to form words
    for each line.
    """
    words = []
    line_height = config.median_height_scalar * Character.median_width
    y_sorted = sorted(characters, key=lambda x: x.centroid[1])

    current_line = []
    for i, char in enumerate(y_sorted):
        if i == 0:
            current_line.append(char)
            continue
        if char.y - current_line[-1].y < line_height:
            current_line.append(char)
        else:
            words.extend(extract_words(current_line))
            current_line = [char]
    words.extend(extract_words(current_line))
    return words

def find_words(characters):
    """
    Get list of words in the image.
    """
    words = extract_text(characters)
    return [''.join([char.classification for char in word]) for word in words]

def classify_characters(characters, model):
    """
    Classify each of the finalized characters.
    """
    final = []
    for char in characters:
        char.classify(model)
        if char.classification != Character.empty:
            final.append(char)
    return final

def get_characters(img):
    """
    Convert image to binary, segment into characters, filter out non-characters, classify characters.
    Return list of character objects including their classifications.
    """
    binary = convert_to_binary(img)
    n_labels, stats, centroids = segment_image(binary)
    characters = filter_characters(img, stats, centroids)
    model = Classifier()
    model.load_model()
    final_chars = classify_characters(characters, model)
    return final_chars

def convert_to_binary(img):
    """
    Convert image to binary using adaptive thresholding on brightness.
    """
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(grey,
                                  255,
                                  cv.ADAPTIVE_THRESH_MEAN_C,
                                  cv.THRESH_BINARY,
                                  config.adaptive_thresh_block_size,
                                  config.adaptive_thresh_constant)
    return thresh

def segment_image(img):
    """
    Segment the image into connected components after applying a morphological opening.
    """
    if config.background_white:
        img = cv.bitwise_not(img)
    opened = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones(config.morph_kernel_size, np.uint8))
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(opened, connectivity=config.connectivity)
    return n_labels, stats, centroids

def filter_characters(image, stats, centroids):
    """
    Filter out characters that are too small or too large relative to the average object size.
    Also, extract the image of each character and initialize a character object.
    :param image: original image
    :param stats: list of stats for each object
    :param centroids: list of centroids for each object
    :return: list of character objects
    """
    characters = []
    # filter out objects that are too small or too large relative avg char size
    avg_area = np.average(stats[1:, 4])
    for i in range(1, len(stats)):
        # skip the background
        x, y, w, h, area = stats[i]
        if 64 < area < avg_area * 2:
            # for grayscale image:
            char_image = image[y:y + h, x:x + w]
            char_image = cv.cvtColor(char_image, cv.COLOR_BGR2GRAY)
            char_image = cv.bitwise_not(char_image)
            characters.append(Character((int(centroids[i][0]), int(centroids[i][1])), x, y, w, h,
                                        Character.scale_char_image(char_image)))
            # for binary image:
            # extract the char window
            # window = np.zeros((h, w), dtype=np.uint8)
            # window[labels[y:y+h, x:x+w] == i] = 1
            # characters.append(Character((int(centroids[i][0]), int(centroids[i][1])), x, y, w, h,
            #                             Character.scale_char_image(window)))

    Character.median_width = np.median([char.width for char in characters])
    Character.median_height = np.median([char.height for char in characters])
    return characters

def bounding_boxes(img, characters):
    """
    Draw bounding boxes around the characters in the image.
    """
    for char in characters:
        x, y = char.x, char.y
        w, h = char.width, char.height
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
