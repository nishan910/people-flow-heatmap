import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

VIDEO_PATH = "data/people-walking.mp4"
OUTPUT_PATH = "output/people_flow_output.mp4"
PERSON_CLASS_ID = 0  # COCO class id for "person"

model = YOLO("yolov8s.pt")
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
W, H = video_info.width, video_info.height

# line Y-coordinates (tune these later using polygonzone.roboflow.com)
LINE_UPPER_Y = int(H * 0.35)
LINE_LOWER_Y = int(H * 0.65)

# dictionary to remember each tracked person's previous center position
track_history = {}   # tracker_id -> (prev_cx, prev_cy)
counted_in_ids = set()
counted_out_ids = set()
in_count = 0
out_count = 0

# heatmap accumulator: one float value per pixel, builds up over the whole video
heatmap_accum = np.zeros((H, W), dtype=np.float32)
last_frame = None

with sv.VideoSink(target_path=OUTPUT_PATH, video_info=video_info) as sink:
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing"):
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == PERSON_CLASS_ID]
        detections = tracker.update_with_detections(detections)

        annotated = frame.copy()

        if detections.tracker_id is not None and len(detections) > 0:
            for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # add this person's center point to the heatmap accumulator
                cv2.circle(heatmap_accum, (cx, cy), radius=18, color=1, thickness=-1)

                if tracker_id in track_history:
                    prev_cx, prev_cy = track_history[tracker_id]

                    # IN: moving downward, crossing the upper line
                    if prev_cy < LINE_UPPER_Y <= cy and tracker_id not in counted_in_ids:
                        in_count += 1
                        counted_in_ids.add(tracker_id)

                    # OUT: moving upward, crossing the lower line
                    if prev_cy > LINE_LOWER_Y >= cy and tracker_id not in counted_out_ids:
                        out_count += 1
                        counted_out_ids.add(tracker_id)

                track_history[tracker_id] = (cx, cy)

        # draw bounding boxes + ID labels
        labels = [f"ID {tid}" for tid in detections.tracker_id] if (
            detections.tracker_id is not None and len(detections) > 0
        ) else []
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # draw the two lines (color coded)
        cv2.line(annotated, (0, LINE_UPPER_Y), (W, LINE_UPPER_Y), (0, 255, 0), 2)  # green = IN line
        cv2.line(annotated, (0, LINE_LOWER_Y), (W, LINE_LOWER_Y), (0, 0, 255), 2)  # red = OUT line

        # live counters
        cv2.putText(annotated, f"IN: {in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"OUT: {out_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        sink.write_frame(frame=annotated)
        last_frame = frame

print(f"Final -> IN: {in_count} | OUT: {out_count}")
print(f"Saved: {OUTPUT_PATH}")

# ---- build the final heatmap ----
heatmap_blurred = cv2.GaussianBlur(heatmap_accum, (0, 0), sigmaX=15, sigmaY=15)
heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

cv2.imwrite("output/heatmap_only.png", heatmap_color)

overlay = cv2.addWeighted(last_frame, 0.5, heatmap_color, 0.5, 0)
cv2.imwrite("output/heatmap_overlay.png", overlay)

print("Saved: output/heatmap_only.png")
print("Saved: output/heatmap_overlay.png")