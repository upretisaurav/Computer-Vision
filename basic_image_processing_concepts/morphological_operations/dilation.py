import cv2
import numpy as np


def demonstrate_dilation():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)

    kernal = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(img, kernal, iterations=1)

    cv2.imshow("Original Image", img)
    cv2.imshow("Dilated Image", dilated)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return dilated


demonstrate_dilation()
