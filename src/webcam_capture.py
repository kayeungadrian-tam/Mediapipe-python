import cv2
import numpy as np
from datetime import datetime

CAMERA_INDEX = 0
FRAME_RATE = 30
SIZE = (640, 480)
OUTPUT_FILE = "./result.mp4"

now = datetime.now()
ts = datetime.timestamp(now)

fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
writer = cv2.VideoWriter(f"./result_{ts}.mp4", fmt, FRAME_RATE, SIZE)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while cap.isOpened():
        ret, frame = cap.read()
        
        writer.write(frame)

        cv2.imshow("Webcam", frame)  # Detect if the Esc key has been pressed
        c = cv2.waitKey(1)
        if c == 27:
            break


    writer.release()
    cap.release()  # Close all active windows

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
