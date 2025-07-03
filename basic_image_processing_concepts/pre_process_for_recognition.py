import cv2
import numpy as np


def preprocess_face_for_recognition(img):
    # Step 1: Convert to grayscale (simplify processing)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Step 2: Denoise the image (reduce noise for better recognition)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Histrogram equalization (normalize lighting)
    equalized = cv2.equalizeHist(denoised)

    # Step 4: Edge enhancement (optional)
    kernal = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpened = cv2.filter2D(equalized, -1, kernal)

    # Step 5: Morphological operations (clean up)
    kernal = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernal)

    cv2.imshow("Preprocessed Image", cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cleaned


preprocess_face_for_recognition(cv2.imread("./images/potrait.jpeg"))
