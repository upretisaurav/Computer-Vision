import mediapipe as mp
import cv2
import numpy as np

# This is just a learning script

class BehaviourTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )
        self.LEFT_EYE = [33, 159, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 386, 263, 373, 374]
        self.EAR_THRESHOLD = 0.35

    def calculate_ear(self, landmarks, eye_points):
        eye_landmarks = np.array([(landmarks[points].x, landmarks[points].y) for points in eye_points])

        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return ear

    def is_eye_closed(self, landmarks, eye_points):
        ear = self.calculate_ear(landmarks, eye_points)
        return ear < self.EAR_THRESHOLD

    def start_tracking(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_closed = self.is_eye_closed(face_landmarks.landmark, self.LEFT_EYE)
                    right_eye_closed = self.is_eye_closed(face_landmarks.landmark, self.RIGHT_EYE)

                    is_sleeping = left_eye_closed and right_eye_closed

                    status = 'SLEEPING' if is_sleeping else 'AWAKE'
                    color = (0, 255, 0) if is_sleeping else (0, 0, 255)
                    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec,
                    )

            cv2.imshow("Behaviour Tracking", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BehaviourTracker()
    tracker.start_tracking()
