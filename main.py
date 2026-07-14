import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/people-walking.mp4"

# Load YOLO model (auto-downloads yolov8n.pt on first run)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()   # read only the first frame
cap.release()

if not ret:
    print("Error: could not read frame.")
else:
    results = model(frame)[0]

    # print detected objects and their confidence scores
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

    # save the annotated frame as an image for visual check
    annotated_frame = results.plot()
    cv2.imwrite("test_frame.jpg", annotated_frame)
    print("Saved: test_frame.jpg")