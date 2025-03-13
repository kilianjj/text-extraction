import util
import cv2 as cv

def main():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord(' '):
            chars = util.get_characters(frame)
            util.bounding_boxes(frame, chars)
            cv.imshow('objects', frame)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
