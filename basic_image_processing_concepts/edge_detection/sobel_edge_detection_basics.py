import cv2
import numpy as np


def demonstrate_sobel_edge_detection():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    soble_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    soble_combined = np.sqrt(sobel_x**2 + soble_y**2)

    soble_combined = np.uint8(np.clip(soble_combined, 0, 255))

    print(f"Sobel _x = {sobel_x}")
    print(f"Sobel _y = {soble_y}")
    print(f"Sobel combined = {soble_combined}")

    return sobel_x, soble_y, soble_combined


demonstrate_sobel_edge_detection()
