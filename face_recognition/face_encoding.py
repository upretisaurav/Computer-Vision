import cv2
import numpy as np
import face_recognition
from PIL import Image
import os


def learn_face_encoding():
    """
    Learn face encoding step by step.
    This converts faces into numerical representations that can be compared.
    """

    print("=== Learning Face Encoding ===")
    print()
    print("WHAT IS FACE ENCODING?")
    print("- Converts a face into a 128-dimensional number array")
    print("- Each number represents different facial features")
    print("- Similar faces will have similar numbers")
    print("- Different faces will have very different numbers")
    print()

    # Step 1: Create a simple test setup
    print("Step 1: Setting up face encoding test...")

    # Start camera to capture face for encoding
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera started successfully.")
    print()
    print("INSTRUCTIONS:")
    print("1. Position your face clearly in the camera")
    print("2. Press SPACE to capture and encode your face")
    print("3. Press 'q' to quit")
    print()

    face_encoding = None
    face_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB (face_recognition library expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw rectangles around faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add instruction text
            cv2.putText(
                frame,
                "Press SPACE to encode this face",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Display status
        if not face_captured:
            cv2.putText(
                frame,
                f"Faces detected: {len(face_locations)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: Capture & Encode | Q: Quit",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Face encoded successfully!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Q: Quit and see encoding details",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Face Encoding Learning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and len(face_locations) > 0 and not face_captured:
            # Capture and encode the first face found
            print("\nCapturing and encoding face...")

            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                face_captured = True

                print("✅ Face encoded successfully!")
                print(f"Encoding is a {len(face_encoding)}-dimensional vector")
                print()

                # Show what the encoding looks like
                print("WHAT DOES AN ENCODING LOOK LIKE?")
                print("Here are the first 10 numbers from your face encoding:")
                for i, value in enumerate(face_encoding[:10]):
                    print(f"  Dimension {i+1}: {value:.6f}")
                print("  ... and 118 more numbers")
                print()

                print("KEY INSIGHTS:")
                print("- Each number captures different facial features")
                print("- Some numbers might represent eye distance, nose shape, etc.")
                print("- The exact meaning is learned by the AI, not programmed")
                print("- These 128 numbers uniquely represent your face!")
                print()
            else:
                print("❌ Could not encode face. Try again.")

    cap.release()
    cv2.destroyAllWindows()

    if face_captured and face_encoding is not None:
        demonstrate_encoding_properties(face_encoding)

    print("Face encoding learning complete!")


def demonstrate_encoding_properties(face_encoding):
    """
    Demonstrate key properties of face encodings
    """
    print("=== Understanding Face Encoding Properties ===")
    print()

    # Property 1: Magnitude
    magnitude = np.linalg.norm(face_encoding)
    print(f"1. ENCODING MAGNITUDE: {magnitude:.6f}")
    print("   - This is like the 'length' of the encoding vector")
    print("   - Usually around 1.0 for normalized encodings")
    print()

    # Property 2: Range of values
    min_val = np.min(face_encoding)
    max_val = np.max(face_encoding)
    mean_val = np.mean(face_encoding)

    print(f"2. VALUE RANGE:")
    print(f"   - Minimum value: {min_val:.6f}")
    print(f"   - Maximum value: {max_val:.6f}")
    print(f"   - Average value: {mean_val:.6f}")
    print("   - Values typically range from -1 to +1")
    print()

    # Property 3: How similarity would be measured
    print("3. HOW SIMILARITY WORKS:")
    print("   - Two face encodings are compared using distance")
    print("   - Smaller distance = more similar faces")
    print("   - Larger distance = more different faces")
    print("   - Threshold (usually 0.6): below = same person, above = different person")
    print()

    # Property 4: Create a slightly modified version to show distance
    print("4. DISTANCE DEMONSTRATION:")
    # Add small random noise to simulate a slightly different photo of same person
    same_person_encoding = face_encoding + np.random.normal(
        0, 0.05, face_encoding.shape
    )
    distance_same = np.linalg.norm(face_encoding - same_person_encoding)

    # Create a random encoding to simulate different person
    different_person_encoding = np.random.normal(0, 0.5, face_encoding.shape)
    different_person_encoding = different_person_encoding / np.linalg.norm(
        different_person_encoding
    )
    distance_different = np.linalg.norm(face_encoding - different_person_encoding)

    print(
        f"   - Distance to slightly modified version (same person): {distance_same:.6f}"
    )
    print(f"   - Distance to random face (different person): {distance_different:.6f}")
    print(f"   - Notice how same person has much smaller distance!")
    print()

    print("NEXT STEPS IN YOUR LEARNING:")
    print("✅ You now understand face encoding")
    print("📋 Next: Learn to compare faces (face matching)")
    print("📋 After that: Build a face database")
    print("📋 Finally: Real-time face recognition")


if __name__ == "__main__":
    print("Face Recognition Learning - Step 1: Face Encoding")
    print("=" * 50)
    print()

    # Check if face_recognition library is installed
    try:
        import face_recognition

        print("✅ face_recognition library is available")
        print()
        learn_face_encoding()
    except ImportError:
        print("❌ face_recognition library not found!")
        print()
        print("INSTALLATION REQUIRED:")
        print("Run this command in your terminal:")
        print("pip install face-recognition")
        print()
        print("Note: This library requires:")
        print("- dlib (for face detection and encoding)")
        print("- cmake (system dependency)")
        print()
        print("If you encounter issues, try:")
        print("brew install cmake  # On Mac")
        print("pip install dlib")
        print("pip install face-recognition")
