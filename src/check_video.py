
import cv2
from pathlib import Path

video_path = Path("data/videos/Adjectives/1. loud/MVI_9448.MOV")

print("Video exists:", video_path.exists())

if video_path.exists():
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: OpenCV could not open the video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("Video opened successfully!")
        print("FPS:", fps)
        print("Total frames:", frame_count)
        print("Resolution:", width, "x", height)

        success, frame = cap.read()
        print("First frame readable:", success)

        cap.release()
