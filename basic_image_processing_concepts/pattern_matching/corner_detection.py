import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_for_corner_detection(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    equalized = cv2.equalizeHist(denoised)

    return equalized


def harris_corner_detection(img_path):
    """Demonstrate Harris Corner Detection step by step"""

    img = cv2.imread(img_path)
    processed_gray = preprocess_for_corner_detection(img)

    print("=== HARRIS CORNER DETECTION ===")
    print("How it works:")
    print("1. Calculates gradients in X and Y directions")
    print("2. Creates a structure tensor (matrix) for each pixel")
    print("3. Analyzes eigenvalues to determine if it's a corner")
    print()

    blockSize = 2
    ksize = 3
    k = 0.04

    harris_response = cv2.cornerHarris(processed_gray, blockSize, ksize, k)

    harris_response = cv2.dilate(harris_response, None)

    img_harris = img.copy()

    thresholds = [0.01, 0.05, 0.1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(harris_response, cmap="hot")
    axes[0, 1].set_title("Harris Response (Heatmap)")
    axes[0, 1].axis("off")

    for i, threshold in enumerate(thresholds):
        img_result = img.copy()

        img_result[harris_response > threshold * harris_response.max()] = [0, 0, 255]

        if i < 2:
            row, col = (1, i)
            axes[row, col].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"Threshold: {threshold} (Red corners)")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

    for threshold in thresholds:
        corner_count = np.sum(harris_response > threshold * harris_response.max())
        print(f"Threshold {threshold}: {corner_count} corners detected")


def compare_corner_methods(img_path):
    """Compare different corner detection methods"""
    img = cv2.imread(img_path)
    processed_gray = preprocess_for_corner_detection(img)

    print("\n=== COMPARING CORNER DETECTION METHODS ===")

    harris_corners = cv2.cornerHarris(processed_gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    img_harris = img.copy()
    img_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    corners_shi_tomasi = cv2.goodFeaturesToTrack(
        processed_gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3
    )

    img_shi_tomasi = img.copy()
    if corners_shi_tomasi is not None:
        corners_shi_tomasi = np.int8(corners_shi_tomasi)
        for corner in corners_shi_tomasi:
            x, y = corner.ravel()
            cv2.circle(img_shi_tomasi, (x, y), 5, (0, 255, 0), -1)

    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
    keypoints_fast = fast.detect(processed_gray, None)
    img_fast = cv2.drawKeypoints(img.copy(), keypoints_fast, None, color=(255, 0, 0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Harris Corners (Red)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(img_shi_tomasi, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f"Shi-Tomasi ({len(corners_shi_tomasi) if corners_shi_tomasi is not None else 0} corners)"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"FAST ({len(keypoints_fast)} corners)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    harris_count = np.sum(harris_corners > 0.01 * harris_corners.max())
    shi_tomasi_count = len(corners_shi_tomasi) if corners_shi_tomasi is not None else 0
    fast_count = len(keypoints_fast)

    print(f"Harris corners: {harris_count}")
    print(f"Shi-Tomasi corners: {shi_tomasi_count}")
    print(f"FAST corners: {fast_count}")

    print("\nCharacteristics:")
    print("Harris: Good quality, slower, gives corner strength")
    print("Shi-Tomasi: Better corner selection, medium speed")
    print("FAST: Very fast, good for real-time applications")


def interactive_corner_tuning(img_path):
    """Interactive corner detection parameter tuning"""
    img = cv2.imread(img_path)
    processed_gray = preprocess_for_corner_detection(img)

    print("\n=== INTERACTIVE PARAMETER TUNING ===")
    print("Adjust parameters to see how they affect corner detection")

    def update_corners(val):

        block_size = cv2.getTrackbarPos("Block Size", "Harris Corners")
        if block_size < 2:
            block_size = 2
        if block_size % 2 == 0:
            block_size += 1

        k_value = cv2.getTrackbarPos("K Value", "Harris Corners") / 1000.0
        threshold = cv2.getTrackbarPos("Threshold", "Harris Corners") / 1000.0

        harris_response = cv2.cornerHarris(processed_gray, block_size, 3, k_value)
        harris_response = cv2.dilate(harris_response, None)

        img_result = img.copy()
        img_result[harris_response > threshold * harris_response.max()] = [0, 0, 255]

        corner_count = np.sum(harris_response > threshold * harris_response.max())

        cv2.putText(
            img_result,
            f"Corners: {corner_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_result,
            f"Block Size: {block_size}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_result,
            f"K: {k_value:.3f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_result,
            f"Threshold: {threshold:.3f}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Harris Corners", img_result)

    cv2.namedWindow("Harris Corners")
    cv2.createTrackbar("Block Size", "Harris Corners", 2, 20, update_corners)
    cv2.createTrackbar("K Value", "Harris Corners", 40, 200, update_corners)
    cv2.createTrackbar("Threshold", "Harris Corners", 10, 100, update_corners)

    update_corners(0)

    print("Use trackbars to adjust parameters. Press 'q' to quit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "../images/potrait.jpeg"

    print("Corner Detection Learning Session")
    print("=" * 40)

    compare_corner_methods(img_path)

    interactive_corner_tuning(img_path)
