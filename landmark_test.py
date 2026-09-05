import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WINDOW = "Landmarks"

def main() -> None:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    wrist = result.hand_landmarks[0][0]
                    print(f"hands: {len(result.hand_landmarks)} wrist x={wrist.x:.3f} y={wrist.y:.3f}")

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