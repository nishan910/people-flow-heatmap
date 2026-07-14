# People Flow Detection using Object Tracking & Heatmap Visualization

Real-time people counting (IN/OUT) using YOLOv8 + ByteTrack, with a movement heatmap.

## Detection Method
- YOLOv8s (`ultralytics`) pre-trained on COCO, filtered to the "person" class only.

## Tracking
- ByteTrack (`supervision` library) assigns a persistent ID to each person.
- A manual dictionary (`track_history`) stores each tracked ID's previous center point.

## Line Coordinates
- Upper line (IN): y = height * 0.35
- Lower line (OUT): y = height * 0.65
- For precise placement, coordinates can be tuned using https://polygonzone.roboflow.com/

## IN/OUT Logic
- IN: previous center was above the upper line, current center is below it (moving downward, crossing the upper line).
- OUT: previous center was below the lower line, current center is above it (moving upward, crossing the lower line).
- Each tracker ID is counted only once per direction.

## Heatmap
- Each frame, every person's bbox center is stamped onto an accumulator array.
- At the end, the accumulator is blurred, normalized, and color-mapped (JET colormap).

## Results
- IN: 11, OUT: 9 (on the sample video)

## How to Run
\`\`\`
pip install -r requirements.txt
python main.py
\`\`\`

## Deliverables
- `main.py` — full pipeline
- `output/people_flow_output.mp4` — annotated output video
- `output/heatmap_only.png`, `output/heatmap_overlay.png` — final heatmap

---

## Project Outputs

You can access the project outputs, processed videos and images in the Google Drive link below:

👉 [Google Drive Link](https://drive.google.com/drive/folders/1sz6MoAr14vHYohwjF4-I8I0aZqoXSSEC?usp=sharing)