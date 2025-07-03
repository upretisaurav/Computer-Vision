import cv2
import numpy as np


def demonstrate_opening():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)

    kernal = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)

    cv2.imshow("Original Image", img)
    cv2.imshow("Opening Image", opening)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return opening


demonstrate_opening()
