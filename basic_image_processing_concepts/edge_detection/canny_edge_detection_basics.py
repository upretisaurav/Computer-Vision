import cv2
import numpy as np


def demonstrate_canny_edge_detection():
    img = cv2.imread("../images/potrait.jpeg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]

    print(f"Image dimensions: {edges.shape}")
    print(f"Data type: {edges.dtype}")
    print(f"Edge pixels found: {edge_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Percentage of edge pixels: {(edge_pixels/total_pixels)*100:.2f}%")

    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges


demonstrate_canny_edge_detection()
