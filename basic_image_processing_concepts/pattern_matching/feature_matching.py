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


def understand_basic_matching(img1_path, img2_path):
    """Understand the basics of feature matching"""
    print("=== BASIC FEATURE MATCHING ===")
    print("Feature matching finds corresponding points between images by:")
    print("1. Detecting features in both images")
    print("2. Computing descriptors for each feature")
    print("3. Comparing descriptors to find matches")
    print("4. Filtering out bad matches")
    print()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = preprocess_for_feature_detection(img1)
    gray2 = preprocess_for_feature_detection(img2)

    sift = cv2.SIFT_create(nfeatures=100)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"Image 1: Found {len(kp1)} keypoints")
    print(f"Image 2: Found {len(kp2)} keypoints")

    if desc1 is not None and desc2 is not None:

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)

        print(f"Found {len(matches)} matches")
        print(f"Best match distance: {matches[0].distance:.2f}")
        print(f"Worst match distance: {matches[-1].distance:.2f}")

        img_matches = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:20],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(
            f"Basic Feature Matching (Top 20 matches)\nGreen lines connect matching features"
        )
        plt.axis("off")
        plt.show()

        distances = [m.distance for m in matches]
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(distances, bins=50, alpha=0.7)
        plt.title("Distribution of Match Distances")
        plt.xlabel("Distance (lower = better match)")
        plt.ylabel("Number of Matches")

        plt.subplot(1, 2, 2)
        plt.plot(range(len(distances)), distances, "b-")
        plt.title("Match Quality (sorted)")
        plt.xlabel("Match Rank")
        plt.ylabel("Distance")
        plt.axhline(y=np.mean(distances), color="r", linestyle="--", label="Average")
        plt.legend()

        plt.tight_layout()
        plt.show()

        return matches, kp1, kp2, desc1, desc2

    return None, kp1, kp2, desc1, desc2


def advanced_matching_techniques(img1_path, img2_path):
    """Compare different matching techniques"""
    print("\n=== ADVANCED MATCHING TECHNIQUES ===")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = preprocess_for_feature_detection(img1)
    gray2 = preprocess_for_feature_detection(img2)

    sift = cv2.SIFT_create(nfeatures=200)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is not None and desc2 is not None:

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches_bf = bf.match(desc1, desc2)
        matches_bf = sorted(matches_bf, key=lambda x: x.distance)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_flann = flann.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches_flann:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Image 1")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Image 2")
        axes[0, 1].axis("off")

        img_bf = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches_bf[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axes[1, 0].imshow(cv2.cvtColor(img_bf, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(
            f"Brute Force Matching\n({len(matches_bf)} total matches, showing top 30)"
        )
        axes[1, 0].axis("off")

        img_ratio = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axes[1, 1].imshow(cv2.cvtColor(img_ratio, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(
            f"FLANN + Ratio Test\n({len(good_matches)} good matches, showing top 30)"
        )
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

        print("Matching Results Comparison:")
        print(f"Brute Force: {len(matches_bf)} matches")
        print(f"FLANN + Ratio Test: {len(good_matches)} matches")

        if len(matches_bf) > 0:
            bf_avg_distance = np.mean([m.distance for m in matches_bf[:50]])
            print(f"Brute Force average distance (top 50): {bf_avg_distance:.2f}")

        if len(good_matches) > 0:
            ratio_avg_distance = np.mean([m.distance for m in good_matches])
            print(f"Ratio Test average distance: {ratio_avg_distance:.2f}")

        print("\nAdvantages:")
        print("Brute Force: Simple, finds all matches, good for small datasets")
        print(
            "FLANN + Ratio: Faster, filters ambiguous matches, better for large datasets"
        )

        return good_matches, kp1, kp2


def homography_and_object_detection(img1_path, img2_path):
    """Use feature matching for object detection with homography"""
    print("\n=== HOMOGRAPHY & OBJECT DETECTION ===")
    print("Homography finds geometric transformation between matched points")
    print("This allows us to locate objects in different images")
    print()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = preprocess_for_feature_detection(img1)
    gray2 = preprocess_for_feature_detection(img2)

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is not None and desc2 is not None and len(desc1) >= 4:

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        print(f"Found {len(good_matches)} good matches")

        if len(good_matches) >= 10:

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:

                h, w = gray1.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

                transformed_corners = cv2.perspectiveTransform(corners, M)

                img2_with_detection = img2.copy()
                cv2.polylines(
                    img2_with_detection,
                    [np.int32(transformed_corners)],
                    True,
                    (0, 255, 0),
                    3,
                )

                inliers = np.sum(mask)
                outliers = len(good_matches) - inliers

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                axes[0, 0].set_title("Template (Object to Find)")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                axes[0, 1].set_title("Scene (Where to Find Object)")
                axes[0, 1].axis("off")

                matchesMask = mask.ravel().tolist()
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=matchesMask,
                    flags=2,
                )

                img_matches = cv2.drawMatches(
                    img1, kp1, img2, kp2, good_matches, None, **draw_params
                )
                axes[1, 0].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title(
                    f"Feature Matches\nGreen = Inliers ({inliers}), Red = Outliers ({outliers})"
                )
                axes[1, 0].axis("off")

                axes[1, 1].imshow(cv2.cvtColor(img2_with_detection, cv2.COLOR_BGR2RGB))
                axes[1, 1].set_title(
                    "Object Detection Result\n(Green box = detected object)"
                )
                axes[1, 1].axis("off")

                plt.tight_layout()
                plt.show()

                print(f"Homography found successfully!")
                print(
                    f"Inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)"
                )
                print(
                    f"Object detected at corners: {transformed_corners.reshape(-1, 2)}"
                )

                return True, M, transformed_corners
            else:
                print("Could not find homography - not enough good matches")
        else:
            print(f"Not enough good matches ({len(good_matches)}) - need at least 10")

    return False, None, None


def interactive_matching_demo(img1_path, img2_path):
    """Interactive demo to understand matching parameters"""
    print("\n=== INTERACTIVE MATCHING DEMO ===")
    print("Adjust parameters to see how they affect matching quality")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = preprocess_for_feature_detection(img1)
    gray2 = preprocess_for_feature_detection(img2)

    def update_matches(val):

        nfeatures = cv2.getTrackbarPos("Features", "Feature Matching")
        if nfeatures < 10:
            nfeatures = 10

        ratio_threshold = (
            cv2.getTrackbarPos("Ratio Threshold", "Feature Matching") / 100.0
        )
        if ratio_threshold < 0.1:
            ratio_threshold = 0.1

        sift = cv2.SIFT_create(nfeatures=nfeatures)
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)

        if desc1 is not None and desc2 is not None:

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)

            img_matches = cv2.drawMatches(
                img1,
                kp1,
                img2,
                kp2,
                good_matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            cv2.putText(
                img_matches,
                f"Features: {nfeatures}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img_matches,
                f"Ratio: {ratio_threshold:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img_matches,
                f"Matches: {len(good_matches)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Feature Matching", img_matches)

    cv2.namedWindow("Feature Matching")
    cv2.createTrackbar("Features", "Feature Matching", 100, 500, update_matches)
    cv2.createTrackbar("Ratio Threshold", "Feature Matching", 70, 100, update_matches)

    update_matches(0)

    print("Adjust parameters and observe:")
    print("- More features = more potential matches but slower")
    print("- Lower ratio threshold = stricter matching, fewer false positives")
    print("- Higher ratio threshold = more matches but more false positives")
    print("Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    img1_path = "../images/potrait.jpeg"
    img2_path = "../images/potrait2.jpeg"

    print("Feature Matching Learning Session")
    print("=" * 50)

    matches, kp1, kp2, desc1, desc2 = understand_basic_matching(img1_path, img2_path)

    if matches is not None:
        good_matches, kp1, kp2 = advanced_matching_techniques(img1_path, img2_path)

        success, homography, corners = homography_and_object_detection(
            img1_path, img2_path
        )

        interactive_matching_demo(img1_path, img2_path)
    else:
        print("Skipping advanced demos - no matches found in basic test")
