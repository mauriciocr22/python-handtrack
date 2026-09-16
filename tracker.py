from operator import index
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import hand_utils

import time

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WINDOW = "Landmarks"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (9, 10), (10, 11), (11, 12),              # middle
    (13, 14), (14, 15), (15, 16),             # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]

def describe (hand: hand_utils.HandState | None) -> str:
    if hand is None:
        return "-"
    return "PINCH" if hand.pinching else "OPEN"

def main() -> None:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            shape_active = False

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

            #    if result.hand_landmarks:
            #        wrist = result.hand_landmarks[0][0]
            #        print(f"hands: {len(result.hand_landmarks)} wrist x={wrist.x:.3f} y={wrist.y:.3f}")
                
                height, width = frame.shape[:2]
                hands = {}

                for hand_index, hand in enumerate(result.hand_landmarks):
                    points = [
                        (int(lm.x * width), int(lm.y * height))
                        for lm in hand
                    ]

                    raw_side = result.handedness[hand_index][0].category_name
                    state = hand_utils.read_hand(points, raw_side)
                    hands[state.side] = state

                    # print(state)

                    label = f"{state.side} {'PINCH' if state.pinching else ''}"
                    cv2.putText(
                        frame, label, points[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    )

                    # for start, end in HAND_CONNECTIONS:   SHOW SKELETON
                    #     cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

                    for index, point in enumerate(points):    
                        cv2.circle(frame, point, 4, (0, 0, 255), -1)
                #        cv2.putText(   SHOW INDEXES
                #            frame,
                #            str(index),
                #            point,
                #            cv2.FONT_HERSHEY_SIMPLEX,
                #            0.3,
                #            (255, 255, 255),
                #            1,
                #        )

                print(hands)

                left = hands.get("Left")
                right = hands.get("Right")

                if left is not None and right is not None:
                    if not shape_active and hand_utils.pinches_touching(left, right):
                        shape_active = True

                if shape_active:
                    shape = hand_utils.build_shape(left, right)
                else:
                    shape = []


                if len(shape) >= 2:
                    for i in range(len(shape)):
                        start = shape[i]
                        end = shape[(i + 1) % len(shape)]
                        cv2.line(frame, start, end, (255, 0, 255), 3)

                for vertex in shape:
                    cv2.circle(frame, vertex, 8, (255, 0, 255), -1)

                status = f"{'ACTIVE' if shape_active else 'IDLE'}"

                cv2.putText(
                    frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
                )

                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()