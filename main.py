import cv2
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

VIDEO_PATH = "data/people-walking.mp4"
OUTPUT_PATH = "output/tracked_output.mp4"
PERSON_CLASS_ID = 0  # COCO class id for "person"

model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)

with sv.VideoSink(target_path=OUTPUT_PATH, video_info=video_info) as sink:
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing"):
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == PERSON_CLASS_ID]
        detections = tracker.update_with_detections(detections)

        labels = [f"ID {tid}" for tid in detections.tracker_id]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        sink.write_frame(frame=annotated)

print(f"Saved: {OUTPUT_PATH}")