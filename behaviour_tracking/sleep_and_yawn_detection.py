import cv2
import mediapipe as mp
import numpy as np
import time


class DebuggingBehaviourTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.LEFT_EYE = [33, 159, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 386, 263, 373, 374]
        self.MOUTH = [61, 291, 39, 181, 0, 17, 18, 200, 269]

        self.eye_closed_frames = 0
        self.mouth_open_frames = 0

        self.sleep_start_time = None
        self.yawn_start_time = None
        self.total_sleep_time = 0
        self.total_yawn_time = 0

        self.sleep_episodes = 0
        self.yawn_episodes = 0

        self.currently_sleeping = False
        self.currently_yawning = False

    def is_eye_closed_debug(self, landmarks, eye_points, eye_name):
        """Debug version with detailed output"""
        eye_coords = []

        for point in eye_points:
            x = landmarks[point].x
            y = landmarks[point].y
            eye_coords.append((x, y))

        eye_coords = np.array(eye_coords)

        if len(eye_coords) >= 6:

            height1 = abs(eye_coords[1][1] - eye_coords[5][1])
            width1 = abs(eye_coords[0][0] - eye_coords[3][0])
            ratio1 = height1 / width1 if width1 > 0 else 0

            height2 = abs(eye_coords[2][1] - eye_coords[4][1])
            width2 = abs(eye_coords[0][0] - eye_coords[3][0])
            ratio2 = height2 / width2 if width2 > 0 else 0

            avg_ratio = (ratio1 + ratio2) / 2

            is_closed = avg_ratio < 0.35

            print(
                f"{eye_name} - Ratio1: {ratio1:.3f}, Ratio2: {ratio2:.3f}, Avg: {avg_ratio:.3f}, Closed: {is_closed}"
            )

            return is_closed
        else:
            return False

    def is_mouth_open(self, landmarks, mouth_points):

        try:
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            left_corner = landmarks[61]
            right_corner = landmarks[291]

            mouth_height = abs(top_lip.y - bottom_lip.y)
            mouth_width = abs(left_corner.x - right_corner.x)

            ratio = mouth_height / mouth_width if mouth_width > 0 else 0

            is_open = ratio > 0.5
            print(
                f"Mouth - Height: {mouth_height:.3f}, Width: {mouth_width:.3f}, Ratio: {ratio:.3f}, Open: {is_open}"
            )

            return is_open
        except:
            return False

    def update_time_counters(self, behaviors):
        """Update time-based counters for behaviors"""
        current_time = time.time()

        if behaviors["sleeping"]:
            if not self.currently_sleeping:
                self.sleep_start_time = current_time
                self.currently_sleeping = True
                self.sleep_episodes += 1
                print(f"🔴 SLEEP EPISODE #{self.sleep_episodes} STARTED")
        else:
            if self.currently_sleeping:
                if self.sleep_start_time:
                    episode_duration = current_time - self.sleep_start_time
                    self.total_sleep_time += episode_duration
                    print(f"🟢 SLEEP EPISODE ENDED - Duration: {episode_duration:.1f}s")
                self.currently_sleeping = False
                self.sleep_start_time = None

        if behaviors["yawning"]:
            if not self.currently_yawning:
                self.yawn_start_time = current_time
                self.currently_yawning = True
                self.yawn_episodes += 1
                print(f"🟡 YAWN EPISODE #{self.yawn_episodes} STARTED")
        else:
            if self.currently_yawning:
                if self.yawn_start_time:
                    episode_duration = current_time - self.yawn_start_time
                    self.total_yawn_time += episode_duration
                    print(f"🟢 YAWN EPISODE ENDED - Duration: {episode_duration:.1f}s")
                self.currently_yawning = False
                self.yawn_start_time = None

    def get_current_sleep_duration(self):
        if self.currently_sleeping and self.sleep_start_time:
            return time.time() - self.sleep_start_time
        return 0

    def get_current_yawn_duration(self):
        if self.currently_yawning and self.yawn_start_time:
            return time.time() - self.yawn_start_time
        return 0

    def draw_debug_info(self, frame):
        """Draw debug information"""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        debug_y = h - 130
        cv2.putText(
            frame,
            f"Eye closed frames: {self.eye_closed_frames}",
            (15, debug_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"Mouth open frames: {self.mouth_open_frames}",
            (15, debug_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"Sleep threshold: {self.eye_closed_frames}/15",
            (15, debug_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"Yawn threshold: {self.mouth_open_frames}/5",
            (15, debug_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    def draw_counters(self, frame):
        """Draw counters in the corner of the frame"""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_start = 25
        line_height = 25

        current_sleep = self.get_current_sleep_duration()
        current_yawn = self.get_current_yawn_duration()

        cv2.putText(
            frame,
            "CURRENT SESSION:",
            (w - 295, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if self.currently_sleeping:
            cv2.putText(
                frame,
                f"Sleeping: {current_sleep:.1f}s",
                (w - 290, y_start + line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Sleeping: 0.0s",
                (w - 290, y_start + line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )

        if self.currently_yawning:
            cv2.putText(
                frame,
                f"Yawning: {current_yawn:.1f}s",
                (w - 290, y_start + 2 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Yawning: 0.0s",
                (w - 290, y_start + 2 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )

        cv2.line(
            frame,
            (w - 295, y_start + 3 * line_height - 5),
            (w - 5, y_start + 3 * line_height - 5),
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "TOTAL STATS:",
            (w - 295, y_start + 3 * line_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"Sleep Episodes: {self.sleep_episodes}",
            (w - 290, y_start + 4 * line_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            frame,
            f"Total Sleep: {self.total_sleep_time:.1f}s",
            (w - 290, y_start + 5 * line_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            frame,
            f"Yawn Episodes: {self.yawn_episodes}",
            (w - 290, y_start + 6 * line_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            frame,
            f"Total Yawns: {self.total_yawn_time:.1f}s",
            (w - 290, y_start + 7 * line_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        return frame

    def analyze_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        behaviors = {"sleeping": False, "yawning": False, "alert": True}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_closed = self.is_eye_closed_debug(
                    face_landmarks.landmark, self.LEFT_EYE, "LEFT"
                )
                right_closed = self.is_eye_closed_debug(
                    face_landmarks.landmark, self.RIGHT_EYE, "RIGHT"
                )

                mouth_open = self.is_mouth_open(face_landmarks.landmark, self.MOUTH)

                if left_closed and right_closed:
                    self.eye_closed_frames += 1
                    print(f"👁️ Both eyes closed - Frame count: {self.eye_closed_frames}")
                else:
                    if self.eye_closed_frames > 0:
                        print(
                            f"👁️ Eyes opened - Resetting frame count from {self.eye_closed_frames}"
                        )
                    self.eye_closed_frames = 0

                if mouth_open:
                    self.mouth_open_frames += 1
                    print(f"👄 Mouth open - Frame count: {self.mouth_open_frames}")
                else:
                    if self.mouth_open_frames > 0:
                        print(
                            f"👄 Mouth closed - Resetting frame count from {self.mouth_open_frames}"
                        )
                    self.mouth_open_frames = 0

                if self.eye_closed_frames > 15:
                    behaviors["sleeping"] = True
                    behaviors["alert"] = False
                    print(f"💤 SLEEPING DETECTED! (frames: {self.eye_closed_frames})")

                if self.mouth_open_frames > 5:
                    behaviors["yawning"] = True
                    print(f"🥱 YAWNING DETECTED! (frames: {self.mouth_open_frames})")

                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    ),
                )

        self.update_time_counters(behaviors)
        frame = self.draw_counters(frame)
        frame = self.draw_debug_info(frame)

        return frame, behaviors


def main():
    tracker = DebuggingBehaviourTracker()
    cap = cv2.VideoCapture(0)

    print("DEBUG MODE - Behavior Tracker")
    print("=" * 50)
    print("Watch the terminal for debug information!")
    print("Thresholds lowered for easier testing:")
    print("- Sleep: 15 frames (was 30)")
    print("- Yawn: 5 frames (was 10)")
    print("- Eye ratio threshold: 0.35 (was 0.25)")
    print("\nPress 'q' to quit, 'r' to reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, behaviors = tracker.analyze_frame(frame)

        y_pos = 30
        for behavior, is_active in behaviors.items():
            color = (0, 255, 0) if is_active else (0, 0, 255)
            text = f'{behavior.upper()}: {"DETECTED" if is_active else "NO"}'
            cv2.putText(
                frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            y_pos += 30

        cv2.imshow("DEBUG - Behavior Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):

            tracker.eye_closed_frames = 0
            tracker.mouth_open_frames = 0
            tracker.total_sleep_time = 0
            tracker.total_yawn_time = 0
            tracker.sleep_episodes = 0
            tracker.yawn_episodes = 0
            tracker.currently_sleeping = False
            tracker.currently_yawning = False
            tracker.sleep_start_time = None
            tracker.yawn_start_time = None
            print("🔄 Counters reset!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
