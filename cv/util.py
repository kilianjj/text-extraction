

import cv2 as cv
import numpy as np
from model import Classifier
import config

# todo: fix scaling
# todo: tune the model and actually get decent accuracy
# todo: upload to GitHub

class Character:
    """
    Class for representing a character.
    """

    empty = ""
    median_width = 0

    def __init__(self, centroid, x, y, w, h, scaled_img, classification=empty):
        """
        Initialize a character with a location, size, image, and classification that empty by default.
        :param centroid: centroid of the character in original image
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
        mapping = {
            0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
            19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'
        }
        return mapping.get(val, '')

    @staticmethod
    def scale_char_image(img):
        """
        Resize the character image to be optimal for classification.
        """
        # todo: handle this better to more accurately preserve aspect ratio of the character
        # todo: pad with zeros on the shortest side to make it square, then resize

        diff = abs(img.shape[0] - img.shape[1])
        if img.shape[0] > img.shape[1]:  # Taller than wide
            pad_left = diff // 2
            pad_right = diff - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant')
        elif img.shape[1] > img.shape[0]:  # Wider than tall
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant')

        w, h = config.model_input_size
        result = np.zeros(config.model_input_size, dtype=np.uint8)
        resized = cv.resize(img, (w - 4, h - 4))
        result[2:h - 2, 2:w - 2] = resized
        return result

    def __str__(self):
        return f'Character {self.classification} at {self.centroid}'

    def is_contiguous(self, other):
        """
        Check if the character is next to the previous one (ordered from top left to bottom right).
        """
        first_end = other.centroid[0] + other.width // 2
        second_start = self.centroid[0] - self.width // 2
        # see if the x distance between them is less than half the median width of chapters
        return abs(first_end - second_start) < config.median_width_scalar * Character.median_width

def extract_text(characters):
    """
    Identify words in the image by grouping contiguous characters together.
    Use relative centroid location to determine if characters are part of the same word.
    """
    words = []
    current_word = []
    for i, char in enumerate(characters):
        if i == 0:
            current_word.append(char)
            continue
        if char.is_contiguous(characters[i - 1]):
            current_word.append(char)
        else:
            words.append(current_word)
            current_word = [char]
    words.append(current_word)
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
    for char in sorted(characters, key=lambda x: (x.centroid[0], x.centroid[1])):
        char.classify(model)
        if char.classification != Character.empty:
            final.append(char)
    return final

def get_characters(img, model_path):
    """
    Convert image to binary, segment into characters, filter out non-characters, classify characters.
    :param img: input image
    :param model_path: path to the trained model
    :return: list of character objects including classification
    """
    binary = convert_to_binary(img)
    n_labels, labels, stats, centroids = segment_image(binary)
    characters = filter_characters(labels, stats, centroids)
    model = Classifier()
    model.load_model(model_path)
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
    return n_labels, labels, stats, centroids

def filter_characters(labels, stats, centroids):
    """
    Filter out characters that are too small or too large relative to the average object size.
    Also, extract the image of each character and initialize a character object.
    :param labels: list of object labels
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
            # extract the char window
            window = np.zeros((h, w), dtype=np.uint8)
            window[labels[y:y+h, x:x+w] == i] = 1
            characters.append(Character((int(centroids[i][0]), int(centroids[i][1])), x, y, w, h,
                                        Character.scale_char_image(window)))
    Character.median_width = np.median([char.width for char in characters])
    return characters

def bounding_boxes(img, characters):
    """
    Draw bounding boxes around the characters in the image.
    """
    for char in characters:
        x, y = char.centroid
        w, h = char.width // 2, char.height // 2
        cv.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 1)
