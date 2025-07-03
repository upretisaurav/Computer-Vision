import datetime
import json
from pathlib import Path
import pickle

import cv2
import face_recognition
import numpy as np


class FaceDatabase:
    """
    A persistent face database that saves and loads face encodings
    """

    def __init__(self, database_path="face_database"):
        self.database_path = Path(database_path)
        self.encodings_file = self.database_path / "face_encodings.pkl"
        self.metadata_file = self.database_path / "face_metadata.json"

        self.database_path.mkdir(exist_ok=True)

        self.known_face_encodings = []
        self.known_face_names = []
        self.face_metadata = {}

        self.load_database()

    def save_database(self):
        """
        Save face encodings and metadata to files
        """

        try:
            # Save encodings as pickle file (Binary format, efficient for numpy arrays)
            with open(self.encodings_file, "wb") as f:
                pickle.dump(
                    {
                        "encodings": self.known_face_encodings,
                        "names": self.known_face_names,
                    },
                    f,
                )

                # Save metadata as JSON file (human-readable format)
                with open(self.metadata_file, "w") as f:
                    json.dump(self.face_metadata, f, indent=2, default=str)

                print(f"Database saved successfully at {self.database_path}")
                print(f" Encodings: {self.encodings_file}")
                print(f" Metadata: {self.metadata_file}")
        except Exception as e:
            print(f"Error saving database: {e}")

    def load_database(self):
        """
        Load face encodings and metadata from files
        """
        try:
            # Load encodings
            if self.encodings_file.exists():
                with open(self.encodings_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get("encodings", [])
                    self.known_face_names = data.get("names", [])
                    print(
                        f"Loaded {len(self.known_face_encodings)} face encodings from {self.encodings_file}"
                    )
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    self.face_metadata = json.load(f)
                    print(
                        f"Loaded metadata for {len(self.face_metadata)} faces from {self.metadata_file}"
                    )

            if len(self.known_face_encodings) == 0:
                print("No existing database found - starting fresh!")

        except Exception as e:
            print(f"Error loading database: {e}")
            print("Starting with an empty database.")

    def add_face(self, face_encoding, name, additional_info=None):
        """
        Add a new face to the database
        """

        # Check if the name already exists
        if name in self.known_face_names:
            print(f"Face with name '{name}' already exists in the database.")
            return

        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

        # Add metadata
        self.face_metadata[name] = {
            "added_data": datetime.datetime.now().isoformat(),
            "face_id": len(self.known_face_names) - 1,
            "additional_info": additional_info or {},
        }

        print(f"Added face for '{name}' to the database.")
        return True

    def remove_face(self, name):
        """
        Remove a face from the database
        """

        if name not in self.known_face_names:
            print(f"{name} not found in the database.")
            return False

        # Find and remove the face
        index = self.known_face_names.index(name)
        self.known_face_encodings.pop(index)
        self.known_face_names.pop(index)

        # Remove metadata
        if name in self.face_metadata:
            del self.face_metadata[name]

        print(f"Removed face for '{name}' from the database.")
        return True

    def find_matches(self, face_encoding, threshold=0.6):
        """
        Find matching faces in the database
        """

        if len(self.known_face_encodings) == 0:
            return [], []

        distances = face_recognition.face_distance(
            self.known_face_encodings, face_encoding
        )
        matches = distances < threshold

        matched_names = []
        matched_distances = []

        for i, (match, distance) in enumerate(zip(matches, distances)):
            if match:
                matched_names.append(self.known_face_names[i])
                matched_distances.append(distance)

        return matched_names, matched_distances

    def get_database_stats(self):
        """
        Get statistics about the database
        """
        return {
            "total_faces": len(self.known_face_encodings),
            "database_path": str(self.database_path),
            "encodings_file_size": (
                self.encodings_file.stat().st_size
                if self.encodings_file.exists()
                else 0
            ),
            "metadata_file_size": (
                self.metadata_file.stat().st_size if self.metadata_file.exists() else 0
            ),
        }

    def list_all_faces(self):
        """
        List all faces in the database with their metadata
        """
        print("\n=== Face Database Contents ===")
        if len(self.known_face_names) == 0:
            print("Database is empty.")
            return

        for i, name in enumerate(self.known_face_names):
            metadata = self.face_metadata.get(name, {})
            added_date = metadata.get("added_data", "Unknown")
            print(f"{i+1}. {name}")
            print(f"    Added: {added_date}")
            print(f"    Additional Info: {metadata.get('additional_info', {})}")
            print()


def learn_persistent_face_database():
    """
    Learn how to build and use a persistent face database
    """

    print("=== Learning Persistent Face Database ===")
    print()
    print("WHAT IS A PERSISTENT FACE DATABASE?")
    print("- Stores face encodings in files on your computer")
    print("- Data survives program restarts")
    print("- Can be shared between different applications")
    print("- Supports adding, removing, and updating faces")
    print("- Includes metadata (when added, additional info, etc.)")
    print()

    # Initialize the database
    db = FaceDatabase("face_database")

    print("DATABASE OPERATIONS:")
    print("1. 'a' - Add new face to database")
    print("2. 'r' - Remove face from database")
    print("3. 'l' - List all faces in database")
    print("4. 't' - Test face recognition")
    print("5. 's' - Show database statistics")
    print("6. 'q' - Quit and save")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mode = "MENU"  # MENU, ADD_FACE, TEST_RECOGNITION

    while True:
        if mode == "MENU":
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            db.list_all_faces()

            choice = input("\n Enter your choice (a/r/l/t/s/q): ").strip().lower()

            if choice == "q":
                db.save_database()
                break
            elif choice == "a":
                mode = "ADD_FACE"
                print("\n📸 ADD FACE MODE")
                print("Position face in camera and press SPACE to capture")
                print("Press ESC to return to menu")
            elif choice == "r":
                if len(db.known_face_names) == 0:
                    print("❌ Database is empty!")
                    continue

                print("\nAvailable faces:")
                for i, name in enumerate(db.known_face_names):
                    print(f"{i+1}. {name}")

                try:
                    selection = input("Enter name to remove: ").strip()
                    db.remove_face(selection)
                    db.save_database()
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
            elif choice == "l":
                db.list_all_faces()
            elif choice == "t":
                if len(db.known_face_encodings) == 0:
                    print("❌ Database is empty! Add some faces first.")
                    continue
                mode = "TEST_RECOGNITION"
                print("\n🎯 FACE RECOGNITION TEST MODE")
                print("Show faces to test recognition")
                print("Press ESC to return to menu")
            elif choice == "s":
                stats = db.get_database_stats()
                print(f"\n📊 DATABASE STATISTICS:")
                print(f"Total faces: {stats['total_faces']}")
                print(f"Database path: {stats['database_path']}")
                print(f"Encodings file size: {stats['encodings_file_size']} bytes")
                print(f"Metadata file size: {stats['metadata_file_size']} bytes")

        elif mode == "ADD_FACE":
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            # Draw rectangles around faces
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Press SPACE to add this face",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"ADD FACE MODE - Faces detected: {len(face_locations)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: Add face | ESC: Back to menu",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Face Database Management", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                mode = "MENU"
                cv2.destroyAllWindows()
            elif key == 32 and len(face_locations) > 0:
                # Add face to database
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )

                if len(face_encodings) > 0:
                    cv2.destroyAllWindows()

                    # Get name from user
                    name = input("\nEnter name for this person: ").strip()
                    if name:
                        # Get additional info
                        role = input("Enter role/description (optional): ").strip()
                        additional_info = {"role": role} if role else {}

                        # Add to database
                        if db.add_face(face_encodings[0], name, additional_info):
                            db.save_database()
                        else:
                            print("❌ Could not add face (name might already exist)")

                    mode = "MENU"

        elif mode == "TEST_RECOGNITION":
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                # Find matches in database
                matched_names, matched_distances = db.find_matches(face_encoding)

                if matched_names:
                    # Get best match
                    best_match_idx = np.argmin(matched_distances)
                    name = matched_names[best_match_idx]
                    distance = matched_distances[best_match_idx]
                    confidence = (1 - distance) * 100

                    color = (0, 255, 0)  # Green
                    label = f"{name} ({confidence:.1f}%)"

                    # Show metadata
                    metadata = db.face_metadata.get(name, {})
                    role = metadata.get("additional_info", {}).get("role", "")
                    if role:
                        label += f" - {role}"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                    label = "Unknown Person"

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

            cv2.putText(
                frame,
                f"RECOGNITION MODE - Database: {len(db.known_face_encodings)} faces",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "ESC: Back to menu",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Face Database Management", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                mode = "MENU"
                cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Final Database Summary ===")
    db.list_all_faces()
    stats = db.get_database_stats()
    print(f"Database stored at: {stats['database_path']}")
    print("Database files:")
    print(f"- face_encodings.pkl ({stats['encodings_file_size']} bytes)")
    print(f"- face_metadata.json ({stats['metadata_file_size']} bytes)")


def demonstrate_database_features():
    """
    Demonstrate advanced database features
    """
    print("\n=== Advanced Database Features ===")
    print()
    print("FILE FORMAT EXPLANATIONS:")
    print()
    print("1. FACE_ENCODINGS.PKL:")
    print("   - Binary file storing numpy arrays (face encodings)")
    print("   - Efficient storage and fast loading")
    print("   - Contains the actual mathematical face representations")
    print()
    print("2. FACE_METADATA.JSON:")
    print("   - Human-readable text file")
    print("   - Contains names, dates, additional information")
    print("   - Easy to edit or view in any text editor")
    print()
    print("BENEFITS OF PERSISTENT DATABASE:")
    print("✅ Data survives program restarts")
    print("✅ Can be backed up and restored")
    print("✅ Sharable between different computers")
    print("✅ Supports large numbers of faces")
    print("✅ Fast loading and searching")
    print("✅ Metadata helps with organization")
    print()


if __name__ == "__main__":
    print("Face Recognition Learning - Step 3: Persistent Face Database")
    print("=" * 65)
    print()
    try:
        learn_persistent_face_database()
        demonstrate_database_features()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")

    print("\nPersistent face database learning complete!")
