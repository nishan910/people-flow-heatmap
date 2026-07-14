import cv2

VIDEO_PATH = "data/people-walking.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Width  : {width}")
    print(f"Video Height : {height}")
    print(f"FPS          : {fps}")
    print(f"Total Frames : {total_frames}")

cap.release()