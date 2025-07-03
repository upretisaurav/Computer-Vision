import cv2
import numpy as np


def dmonstrate_closing():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)

    kernal = np.ones((5, 5), np.uint8)

    dmonstrate_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)

    cv2.imshow("Original Image", img)
    cv2.imshow("Closing Image", dmonstrate_closing)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return dmonstrate_closing


dmonstrate_closing()
