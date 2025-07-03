import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_for_feature_detection(img):
    """Same preprocessing as before"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.equalizeHist(denoised)
    return equalized


def understand_sift_descriptors(img_path):
    """Deep dive into SIFT descriptors"""
    img = cv2.imread(img_path)
    gray = preprocess_for_feature_detection(img)

    print("=== SIFT FEATURE DESCRIPTORS ===")
    print("SIFT creates 128-dimensional descriptors that are:")
    print("1. Scale invariant (same object at different sizes)")
    print("2. Rotation invariant (same object rotated)")
    print("3. Partially illumination invariant")
    print()

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=50)  # Limit to 50 features for clarity

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print(f"Found {len(keypoints)} keypoints")
    print(f"Each descriptor has {descriptors.shape[1]} dimensions")
    print(f"Descriptor data type: {descriptors.dtype}")
    print()

    # Visualize keypoints with different information
    img_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show descriptor values for first few keypoints
    print("Sample descriptor values (first 3 keypoints):")
    for i in range(min(3, len(descriptors))):
        print(f"Keypoint {i}: {descriptors[i][:10]}... (showing first 10 values)")
        print(f"  Position: ({keypoints[i].pt[0]:.1f}, {keypoints[i].pt[1]:.1f})")
        print(f"  Scale: {keypoints[i].size:.1f}")
        print(f"  Angle: {keypoints[i].angle:.1f}°")
        print()

    # Visualize descriptor as histogram
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image with keypoints
    axes[0, 0].imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("SIFT Keypoints\n(Size = scale, Arrow = orientation)")
    axes[0, 0].axis("off")

    # Descriptor histogram for first keypoint
    if len(descriptors) > 0:
        axes[0, 1].bar(range(128), descriptors[0])
        axes[0, 1].set_title("Descriptor Vector (Keypoint 0)")
        axes[0, 1].set_xlabel("Dimension")
        axes[0, 1].set_ylabel("Value")

    # Descriptor heatmap for multiple keypoints
    if len(descriptors) >= 10:
        im = axes[1, 0].imshow(descriptors[:10], cmap="viridis", aspect="auto")
        axes[1, 0].set_title("First 10 Descriptors (Heatmap)")
        axes[1, 0].set_xlabel("Descriptor Dimension")
        axes[1, 0].set_ylabel("Keypoint Index")
        plt.colorbar(im, ax=axes[1, 0])

    # Show how descriptors change with rotation
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_gray = preprocess_for_feature_detection(rotated_img)
    kp_rot, desc_rot = sift.detectAndCompute(rotated_gray, None)

    img_rotated_kp = cv2.drawKeypoints(
        rotated_img, kp_rot, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    axes[1, 1].imshow(cv2.cvtColor(img_rotated_kp, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Rotated Image\n({len(kp_rot)} keypoints)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    return keypoints, descriptors


def compare_descriptor_methods(img_path):
    """Compare SIFT vs ORB descriptors"""
    img = cv2.imread(img_path)
    gray = preprocess_for_feature_detection(img)

    print("\n=== COMPARING DESCRIPTOR METHODS ===")

    # SIFT descriptors
    sift = cv2.SIFT_create(nfeatures=100)
    kp_sift, desc_sift = sift.detectAndCompute(gray, None)

    # ORB descriptors
    orb = cv2.ORB_create(nfeatures=100)
    kp_orb, desc_orb = orb.detectAndCompute(gray, None)

    # Draw keypoints
    img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 255, 0))
    img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(255, 0, 0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # SIFT keypoints
    axes[0, 1].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"SIFT Keypoints ({len(kp_sift)})")
    axes[0, 1].axis("off")

    # ORB keypoints
    axes[0, 2].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"ORB Keypoints ({len(kp_orb)})")
    axes[0, 2].axis("off")

    # SIFT descriptor visualization
    if desc_sift is not None and len(desc_sift) > 0:
        axes[1, 0].bar(range(128), desc_sift[0])
        axes[1, 0].set_title("SIFT Descriptor (128D)")
        axes[1, 0].set_xlabel("Dimension")

    # ORB descriptor visualization
    if desc_orb is not None and len(desc_orb) > 0:
        axes[1, 1].bar(range(32), desc_orb[0])
        axes[1, 1].set_title("ORB Descriptor (32D)")
        axes[1, 1].set_xlabel("Dimension")

    # Comparison table
    axes[1, 2].axis("off")
    comparison_text = """
    COMPARISON:

    SIFT:
    • 128 dimensions
    • Float values
    • Very robust
    • Slower computation
    • Scale + rotation invariant

    ORB:
    • 32 dimensions
    • Binary values (0/1)
    • Good performance
    • Very fast
    • Rotation invariant
    """
    axes[1, 2].text(
        0.1,
        0.9,
        comparison_text,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue"),
    )

    plt.tight_layout()
    plt.show()

    # Print technical details
    print(
        f"SIFT: {len(kp_sift)} keypoints, descriptor shape: {desc_sift.shape if desc_sift is not None else 'None'}"
    )
    print(
        f"ORB: {len(kp_orb)} keypoints, descriptor shape: {desc_orb.shape if desc_orb is not None else 'None'}"
    )

    if desc_sift is not None:
        print(f"SIFT descriptor range: {desc_sift.min():.2f} to {desc_sift.max():.2f}")
    if desc_orb is not None:
        print(f"ORB descriptor range: {desc_orb.min()} to {desc_orb.max()}")


def descriptor_invariance_test(img_path):
    """Test how descriptors handle transformations"""
    img = cv2.imread(img_path)
    gray = preprocess_for_feature_detection(img)

    print("\n=== TESTING DESCRIPTOR INVARIANCE ===")

    # Original image
    sift = cv2.SIFT_create(nfeatures=50)
    kp_orig, desc_orig = sift.detectAndCompute(gray, None)

    # Create transformations
    height, width = gray.shape

    # 1. Scaled image (50% smaller)
    scaled = cv2.resize(gray, (width // 2, height // 2))
    kp_scaled, desc_scaled = sift.detectAndCompute(scaled, None)

    # 2. Rotated image (45 degrees)
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(gray, rotation_matrix, (width, height))
    kp_rotated, desc_rotated = sift.detectAndCompute(rotated, None)

    # 3. Brightness changed
    bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    kp_bright, desc_bright = sift.detectAndCompute(bright, None)

    # Visualize all transformations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    transformations = [
        (img, kp_orig, "Original"),
        (
            cv2.cvtColor(
                cv2.resize(
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (width // 2, height // 2)
                ),
                cv2.COLOR_BGR2RGB,
            ),
            kp_scaled,
            "Scaled (50%)",
        ),
        (
            cv2.cvtColor(
                cv2.warpAffine(
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    rotation_matrix,
                    (width, height),
                ),
                cv2.COLOR_BGR2RGB,
            ),
            kp_rotated,
            "Rotated (45°)",
        ),
        (
            cv2.cvtColor(
                cv2.convertScaleAbs(
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), alpha=1.5, beta=30
                ),
                cv2.COLOR_BGR2RGB,
            ),
            kp_bright,
            "Brighter",
        ),
    ]

    for i, (img_show, kp, title) in enumerate(transformations):
        # Draw keypoints
        if i == 0:
            img_with_kp = cv2.drawKeypoints(
                cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR), kp, None, color=(0, 255, 0)
            )
            axes[0, i].imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        else:
            img_with_kp = cv2.drawKeypoints(
                cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR), kp, None, color=(0, 255, 0)
            )
            axes[0, i].imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))

        axes[0, i].set_title(f"{title}\n({len(kp)} keypoints)")
        axes[0, i].axis("off")

    # Show descriptor similarity (first descriptor comparison)
    descriptors = [desc_orig, desc_scaled, desc_rotated, desc_bright]
    titles = ["Original", "Scaled", "Rotated", "Brighter"]

    for i, (desc, title) in enumerate(zip(descriptors, titles)):
        if desc is not None and len(desc) > 0:
            axes[1, i].bar(
                range(min(64, len(desc[0]))), desc[0][:64]
            )  # Show first 64 dimensions
            axes[1, i].set_title(f"{title} Descriptor")
            axes[1, i].set_xlabel("Dimension (first 64)")
        else:
            axes[1, i].text(0.5, 0.5, "No descriptors", ha="center", va="center")
            axes[1, i].set_title(f"{title} - No features")

    plt.tight_layout()
    plt.show()

    # Calculate descriptor similarities
    if all(desc is not None and len(desc) > 0 for desc in descriptors):
        print("Descriptor Analysis:")
        print("(Comparing first descriptor from each transformation)")

        base_desc = desc_orig[0]
        for desc, title in zip(descriptors[1:], titles[1:]):
            if len(desc) > 0:
                # Calculate cosine similarity
                similarity = np.dot(base_desc, desc[0]) / (
                    np.linalg.norm(base_desc) * np.linalg.norm(desc[0])
                )
                print(f"Similarity with {title}: {similarity:.3f}")


def interactive_descriptor_explorer(img_path):
    """Interactive exploration of descriptors"""
    img = cv2.imread(img_path)
    gray = preprocess_for_feature_detection(img)

    print("\n=== INTERACTIVE DESCRIPTOR EXPLORER ===")
    print("Click on the image to see descriptor at that location")

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Create image with keypoints
    img_with_kp = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest keypoint
            min_dist = float("inf")
            closest_kp_idx = -1

            for i, kp in enumerate(keypoints):
                dist = np.sqrt((kp.pt[0] - x) ** 2 + (kp.pt[1] - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_kp_idx = i

            if closest_kp_idx != -1 and min_dist < 20:  # Within 20 pixels
                kp = keypoints[closest_kp_idx]
                desc = descriptors[closest_kp_idx]

                print(f"\nKeypoint {closest_kp_idx}:")
                print(f"  Position: ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
                print(f"  Size: {kp.size:.1f}")
                print(f"  Angle: {kp.angle:.1f}°")
                print(f"  Response: {kp.response:.3f}")
                print(f"  Descriptor: {desc[:10]}... (first 10 values)")

                # Highlight selected keypoint
                img_highlight = img_with_kp.copy()
                cv2.circle(
                    img_highlight, (int(kp.pt[0]), int(kp.pt[1])), 15, (0, 0, 255), 3
                )
                cv2.imshow("SIFT Descriptors", img_highlight)

    cv2.namedWindow("SIFT Descriptors")
    cv2.setMouseCallback("SIFT Descriptors", mouse_callback)
    cv2.imshow("SIFT Descriptors", img_with_kp)

    print("Click on green circles to see descriptor details. Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


# Test functions
if __name__ == "__main__":
    img_path = "../images/potrait.jpeg"

    print("Feature Descriptors Learning Session")
    print("=" * 50)

    # Step 1: Understand SIFT descriptors
    understand_sift_descriptors(img_path)

    # Step 2: Compare different descriptor methods
    compare_descriptor_methods(img_path)

    # Step 3: Test invariance properties
    descriptor_invariance_test(img_path)

    # Step 4: Interactive exploration
    interactive_descriptor_explorer(img_path)
