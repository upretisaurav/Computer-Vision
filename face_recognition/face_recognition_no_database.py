import cv2
import numpy as np
import face_recognition
from PIL import Image
import os


def learn_face_matching():
    """
    Learn face matching step by step.
    This compares face encodings to determine if faces match.
    """

    print("=== Learning Face Matching ===")
    print()
    print("WHAT IS FACE MATCHING?")
    print("- Compares two face encodings to see if they're the same person")
    print("- Uses mathematical distance between encoding vectors")
    print("- Returns True/False based on a similarity threshold")
    print("- Threshold typically 0.6: below = match, above = no match")
    print()

    # Storage for reference faces
    known_face_encodings = []
    known_face_names = []

    print("Step 1: Building your reference face database...")
    print()
    print("INSTRUCTIONS:")
    print("1. We'll capture reference faces first")
    print("2. Then test matching against new faces")
    print("3. Press SPACE to capture a reference face")
    print("4. Press 'n' when done adding faces")
    print("5. Press 'q' to quit anytime")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Phase 1: Capture reference faces
    phase = "CAPTURE"  # CAPTURE, MATCH
    face_count = 0

    while phase == "CAPTURE":
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw rectangles around faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Press SPACE to save as reference",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Display status
        cv2.putText(
            frame,
            f"CAPTURE MODE - Reference faces: {len(known_face_encodings)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Faces detected: {len(face_locations)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "SPACE: Save face | N: Start matching | Q: Quit",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Matching Learning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(" ") and len(face_locations) > 0:
            # Capture reference face
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) > 0:
                face_count += 1
                name = f"Person_{face_count}"
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)

                print(f"✅ Captured reference face: {name}")
                print(f"   Total reference faces: {len(known_face_encodings)}")

                # Show encoding comparison
                if len(known_face_encodings) > 1:
                    distance = face_recognition.face_distance(
                        [known_face_encodings[-2]], known_face_encodings[-1]
                    )[0]
                    print(f"   Distance to previous face: {distance:.4f}")
                    if distance < 0.6:
                        print("   ⚠️  These faces seem similar (distance < 0.6)")
                    else:
                        print("   ✅ These faces are different (distance > 0.6)")
                print()

        elif key == ord("n"):
            if len(known_face_encodings) > 0:
                phase = "MATCH"
                print(
                    f"\n🎯 Starting matching phase with {len(known_face_encodings)} reference faces!"
                )
                print()
                print("MATCHING PHASE INSTRUCTIONS:")
                print("- Show your face or ask someone else to try")
                print("- The system will identify if it matches any reference face")
                print("- Watch the distance values and match results")
                print("- Press 'r' to return to capture mode")
                print("- Press 'q' to quit")
                print()
            else:
                print("❌ Please capture at least one reference face first!")

    # Phase 2: Face matching
    while phase == "MATCH":
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Compare with all known faces
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            # Determine if it's a match
            threshold = 0.6
            is_match = best_distance < threshold

            if is_match:
                name = known_face_names[best_match_index]
                confidence = (1 - best_distance) * 100
                color = (0, 255, 0)  # Green for match
                label = f"{name} ({confidence:.1f}%)"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for no match
                label = f"{name} (d={best_distance:.3f})"

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Show distance to all known faces (for learning)
            y_offset = top - 10
            for i, (known_name, distance) in enumerate(
                zip(known_face_names, face_distances)
            ):
                match_status = "✓" if distance < threshold else "✗"
                distance_text = f"{match_status} {known_name}: {distance:.3f}"
                cv2.putText(
                    frame,
                    distance_text,
                    (left, y_offset - (i * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Display matching info
        cv2.putText(
            frame,
            f"MATCHING MODE - References: {len(known_face_encodings)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Threshold: {threshold} (lower = more similar)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "R: Return to capture | Q: Quit",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Face Matching Learning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            phase = "CAPTURE"
            print("\n📸 Returned to capture mode")
            print("Add more reference faces or press 'n' to return to matching")
            print()

    cap.release()
    cv2.destroyAllWindows()

    # Show final analysis
    if len(known_face_encodings) > 0:
        analyze_face_database(known_face_encodings, known_face_names)


def analyze_face_database(encodings, names):
    """
    Analyze the captured face database to show distances between all faces
    """
    print("\n=== Face Database Analysis ===")
    print()
    print("DISTANCE MATRIX:")
    print("This shows how similar/different your reference faces are to each other")
    print()

    # Create distance matrix
    print("Distance between faces:")
    print("-" * 50)

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:  # Only show upper triangle to avoid duplicates
                distance = face_recognition.face_distance([encodings[i]], encodings[j])[
                    0
                ]
                similarity = (1 - distance) * 100 if distance < 1 else 0

                print(f"{name1} ↔ {name2}:")
                print(f"  Distance: {distance:.4f}")
                print(f"  Similarity: {similarity:.1f}%")

                if distance < 0.6:
                    print(f"  Status: ✅ Would be considered SAME PERSON")
                else:
                    print(f"  Status: ❌ Would be considered DIFFERENT PEOPLE")
                print()

    print("KEY LEARNINGS:")
    print("✅ Face matching uses distance between encodings")
    print("✅ Threshold 0.6 is the typical decision boundary")
    print("✅ Lower distances = more similar faces")
    print("✅ You can adjust threshold based on your needs:")
    print("   - Lower threshold (0.4) = stricter matching")
    print("   - Higher threshold (0.8) = more lenient matching")
    print()

    print("REAL-WORLD APPLICATIONS:")
    print("- Door access control systems")
    print("- Photo organization (group by person)")
    print("- Security cameras with person identification")
    print("- Attendance systems")
    print()

    print("NEXT STEPS IN YOUR LEARNING:")
    print("✅ You now understand face encoding")
    print("✅ You now understand face matching")
    print("📋 Next: Build a persistent face database")
    print("📋 After that: Real-time face recognition system")


def demonstrate_threshold_effects():
    """
    Create a small demo showing how different thresholds affect matching
    """
    print("\n=== Understanding Threshold Effects ===")

    # Simulate different distance scenarios
    scenarios = [
        ("Identical photo", 0.0),
        ("Same person, different lighting", 0.3),
        ("Same person, different angle", 0.5),
        ("Borderline case", 0.6),
        ("Similar looking people", 0.8),
        ("Completely different people", 1.2),
    ]

    thresholds = [0.4, 0.6, 0.8]

    print("How different thresholds affect matching decisions:")
    print("-" * 60)
    print(
        f"{'Scenario':<30} {'Distance':<10} {'Threshold 0.4':<12} {'Threshold 0.6':<12} {'Threshold 0.8':<12}"
    )
    print("-" * 60)

    for scenario, distance in scenarios:
        results = []
        for threshold in thresholds:
            match = "MATCH" if distance < threshold else "NO MATCH"
            results.append(match)

        print(
            f"{scenario:<30} {distance:<10.1f} {results[0]:<12} {results[1]:<12} {results[2]:<12}"
        )

    print("-" * 60)
    print()
    print("THRESHOLD SELECTION TIPS:")
    print("- 0.4: Very strict - reduces false positives but may miss valid matches")
    print("- 0.6: Balanced - good for most applications")
    print("- 0.8: Lenient - catches more matches but may have false positives")


if __name__ == "__main__":
    print("Face Recognition Learning - Step 2: Face Matching")
    print("=" * 55)
    print()

    try:
        learn_face_matching()
        demonstrate_threshold_effects()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")

    print("\nFace matching learning complete!")
