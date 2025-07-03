import cv2
import numpy as np


def demonstrate_erosion():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)

    kernal = np.ones((5, 5), np.uint8)

    eroded = cv2.erode(img, kernel=kernal, iterations=1)

    cv2.imshow("Original Image", img)
    cv2.imshow("Eroded Image", eroded)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return eroded


demonstrate_erosion()
