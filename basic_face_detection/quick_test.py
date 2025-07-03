import cv2
import numpy as np


def simple_face_detection_demo():
    """
    Simple face detection demo using Haar Cascades
    This shows the DETECTION part of face recognition
    """
    print("Face Detection Demo")
    print("This will detect faces in real-time using Haar Cascades")
    print("Press 'q' to quit")

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Optional: eye detector for more accuracy
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale (Haar cascades work on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # How much the image size is reduced at each scale
            minNeighbors=5,  # How many neighbors each candidate rectangle should have to retain it
            minSize=(30, 30),  # Minimum possible face size
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        print(f"Detected {len(faces)} face(s)")

        # Draw rectangles around faces
        for x, y, w, h in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add label
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

            # Detect eyes within the face region
            roi_color = frame[y : y + h, x : x + w]
            roi_gray = gray[y : y + h, x : x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Face Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def manual_distance_demo():
    """
    Manual implementation of simple face recognition using template matching
    This shows the RECOGNITION theory without opencv-contrib
    """

    print("\nManual Face Recognition Demo")
    print("This will capture a reference image and then try to match it")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    reference_face = None

    print("Step 1: Capture reference face - Press SPACE when ready")

    while reference_face is None:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Press SPACE to capture reference",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capture Reference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and len(faces) > 0:
            # Capture the largest face as reference
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            reference_face = cv2.resize(gray[y : y + h, x : x + w], (100, 100))
            print("Reference face captured!")
            break

    if reference_face is None:
        print("No reference face captured")
        cap.release()
        cv2.destroyAllWindows()
        return

    print("Step 2: Now testing recognition - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for x, y, w, h in faces:
            # Extract and resize face
            test_face = cv2.resize(gray[y : y + h, x : x + w], (100, 100))

            # Calculate similarity using normalized cross-correlation
            # Simple template matching approach
            result = cv2.matchTemplate(test_face, reference_face, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]

            # Threshold for recognition (adjust as needed)
            if similarity > 0.6:
                color = (0, 255, 0)  # Green for match
                label = f"MATCH ({similarity:.2f})"
            else:
                color = (0, 0, 255)  # Red for no match
                label = f"NO MATCH ({similarity:.2f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Face Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose demo:")
    print("1. Face Detection Demo")
    print("2. Simple Face Recognition Demo")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        simple_face_detection_demo()
    elif choice == "2":
        manual_distance_demo()
    else:
        print("Invalid choice")
