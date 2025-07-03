import cv2
import cv2.data


def learn_face_detection():
    """
    Learn face detection step by step.
    This is the foundation of all face recognition systems
    """

    print("=== Learning Face Detection ===")
    print()
    print("WHAT IS FACE DETECTION?")
    print("- Finding WHERE faces are located in an image")
    print("- Returns coordinates: (x, y, width, height)")
    print("- Does not identify who the person is")
    print()

    # Step 1: Load the face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("Face detector loaded successfully.")
    print()

    print("WHAT IS HAAR CASCADE?")
    print("- Pre-trained model that learned to recognize face patterns")
    print("- Uses simple rectangular features (light/dark regions)")
    print("- Fast and efficient for real-time detection")
    print()

    # Step 2: Start camera
    print("Starting camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera started successfully.")
    print()
    print("CONTROLS:")
    print("- Look at the camera")
    print("- Press 'q' to quit")
    print()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        print(f"This is ret: {ret}")
        print(f"This is frame: {frame}")
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        # Step 4: Convert to grayscale
        # WHY? Haar cascades work on intensity (brightness) not color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 5: Detect faces
        faces = face_cascade.detectMultiScale(
            gray,  # Input image (grayscale)
            scaleFactor=1.1,  # How much to reduce image size at each scale (1.1 = 10% smaller)
            minNeighbors=5,  # How many neighbors each face needs (reduces false positives)
            minSize=(30, 30),  # Minimum face size in pixel
        )

        # Step 6: Print what we found (every 30 frames to avoid spam)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Detected {len(faces)} face(s)")

            # Show details of each face
            for i, (x, y, w, h) in enumerate(faces):
                print(f"  Face {i+1}: position=({x},{y}), size={w}x{h} pixels")

        # Step 7: Draw rectangles around detected faces
        for x, y, w, h in faces:
            # Draw blue rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add label above face
            cv2.putText(
                frame,
                f"Face {w}x{h}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Step 8: Show information on screen
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Step 9: Display the frame
        cv2.imshow("Learning Face Detection", frame)

        # Step 10: Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting...")
            break

    # Step 11: Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed, windows closed")
    print("Face detection learning complete!")


def experiment_with_parameters():
    """
    Experiment with different detection parameters to understand their effects
    """
    print("\n=== EXPERIMENTING WITH DETECTION PARAMETERS ===")
    print("This will show you how different settings affect face detection")
    print()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    # Different parameter sets to try
    parameter_sets = [
        {
            "name": "Very Sensitive",
            "scaleFactor": 1.05,
            "minNeighbors": 3,
            "minSize": (20, 20),
        },
        {
            "name": "Balanced",
            "scaleFactor": 1.1,
            "minNeighbors": 5,
            "minSize": (30, 30),
        },
        {
            "name": "Very Strict",
            "scaleFactor": 1.3,
            "minNeighbors": 8,
            "minSize": (50, 50),
        },
    ]

    current_params = 0

    print("CONTROLS:")
    print("- Press SPACE to cycle through different parameter sets")
    print("- Press 'q' to quit")
    print("- Watch how the number of detected faces changes!")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use current parameter set
        params = parameter_sets[current_params]
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=params["scaleFactor"],
            minNeighbors=params["minNeighbors"],
            minSize=params["minSize"],
        )

        # Draw faces
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show current settings
        cv2.putText(
            frame,
            f"Mode: {params['name']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"ScaleFactor: {params['scaleFactor']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"MinNeighbors: {params['minNeighbors']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "SPACE: change mode",
            (10, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Parameter Experiment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            current_params = (current_params + 1) % len(parameter_sets)
        print(f"Current mode: {params['name']} - Detected {len(faces)} face(s)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Face Detection Learning")
    print("=" * 30)
    print()
    print("What would you like to learn?")
    print("1. Learn Basic Face Detection")
    print("2. Experiment with Parameters")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        learn_face_detection()
    elif choice == "2":
        experiment_with_parameters()
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
