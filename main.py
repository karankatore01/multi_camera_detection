import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from reid_model import extract_embedding
from database import get_all_persons, add_or_update_person
from utils import find_best_match
from config import (
    FRAME_SKIP,
    SIMILARITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MAX_EMBEDDINGS_PER_PERSON
)

# Load models
detector = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=60)

# Two camera videos
video_sources = [
    r"C:\Users\karan\Downloads\reid_inputs\BT1301.mp4",
    r"C:\Users\karan\Downloads\reid_inputs\BT801.mp4"
]

caps = [cv2.VideoCapture(v) for v in video_sources]
frame_counts = [0] * len(caps)

while True:
    for cam_id, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        # frame skipping PER CAMERA
        frame_counts[cam_id] += 1
        if frame_counts[cam_id] % FRAME_SKIP != 0:
            continue

        results = detector(frame, conf=CONFIDENCE_THRESHOLD)[0]
        detections = []

        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1],
                     float(box.conf[0]),
                     "person")
                )

        tracks = tracker.update_tracks(detections, frame=frame)
        persons = get_all_persons()   # DB call once per frame

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            
            crop = frame[t:b, l:r]
            if crop.size == 0:
                continue

            embedding = extract_embedding(crop)

            best_id, best_score = find_best_match(embedding, persons)

            if best_score < SIMILARITY_THRESHOLD:
                best_id = None

            global_id = add_or_update_person(
                best_id,
                embedding,
                MAX_EMBEDDINGS_PER_PERSON
            )

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {global_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(f"Camera {cam_id}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
