import cv2
from matplotlib import pyplot as plt
import numpy as np
from mtcnn import MTCNN
from skimage.feature import hog


def hist_equilise(img):
    img = cv2.equalizeHist(img)
    return img


def hog_transform(img):
    hog_features = hog(img, block_norm="L2-Hys", pixels_per_cell=(16, 16))
    return hog_features


def local_face_points(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    result = mtcnn.detect_faces(img)
    if len(result) == 0:
        return None
    for face in result:
        left_eye = list(face["keypoints"]["left_eye"])
        right_eye = list(face["keypoints"]["right_eye"])
        nose = list(face["keypoints"]["nose"])
        mouth_left = list(face["keypoints"]["mouth_left"])
        mouth_right = list(face["keypoints"]["mouth_right"])
    return [left_eye, right_eye, nose, mouth_left, mouth_right]


def filter_edge(img):

    sobel_64 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    abs_64 = np.absolute(sobel_64)
    sobel_8u = np.uint8(abs_64)
    return img


if __name__ == "__main__":
    image = cv2.imread("face.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = hist_equilise(gray)
    filter_outptut = filter_edge(hist)
    cv2.imshow("gray", gray)
    cv2.imshow("hist", hist)
    cv2.imshow("filter_output", filter_outptut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
