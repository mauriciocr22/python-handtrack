import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


import cv2

WINDOW = "Webcam"

def main() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try index 1 or 2.")

    try:
        while True:
            ok, frame = cap.read()
            print(frame.shape)
            if not ok:
                print("Dropped frame")
                continue

            frame = cv2.flip(frame, 1)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break


            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
#          if cv2.waitKey(1) & 0xFF == ord("q"):
#                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()