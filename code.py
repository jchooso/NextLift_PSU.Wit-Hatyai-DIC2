from ultralytics import YOLO
import cv2
import numpy as np
import sys
from collections import deque, Counter

VIDEO_PATH_CABIN = '...'
VIDEO_PATH_WAITING = '...'

MAX_CAPACITY = 6
CONF_THRESHOLD = 0.5
LARGE_PERSON_AREA_THRESHOLD = 2000
SEQUENCE_LENGTH = 10
SKIP_FRAMES = 3 

status_buffer = deque(maxlen=SEQUENCE_LENGTH)

try:
    model = YOLO('yolo12n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit()

cap1 = cv2.VideoCapture(VIDEO_PATH_CABIN)
cap2 = cv2.VideoCapture(VIDEO_PATH_WAITING)

if not cap1.isOpened() or not cap2.isOpened():
    print(f"Error: Cannot open video source.")
    sys.exit()

print("Starting System...")

frame_counter = 0
cached_boxes = None

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    target_h = 480
    target_w = 640
    frame1 = cv2.resize(frame1, (target_w, target_h))
    frame2 = cv2.resize(frame2, (target_w, target_h))
    frame = np.hstack((frame1, frame2))

    h, w, _ = frame.shape

    CABIN_ZONE_X_END = w // 2
    WAITING_ZONE_X_START = w // 2
    roi_margin_x = int(w * 0.05)
    roi_margin_y = int(h * 0.1)

    COUNTING_ROI_RECT = (
        roi_margin_x,
        roi_margin_y,
        CABIN_ZONE_X_END - int(target_w * 0.1),
        h - roi_margin_y
    )

    if frame_counter % SKIP_FRAMES == 0:
        results = model(frame, verbose=False, imgsz=640)
        cached_boxes = results[0].boxes

    current_frame_count = 0
    waiting_count = 0

    cv2.line(frame, (CABIN_ZONE_X_END, 0), (CABIN_ZONE_X_END, h), (255, 255, 255), 4)
    cv2.putText(frame, "CABIN ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "WAITING ZONE", (WAITING_ZONE_X_START + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.rectangle(frame, (COUNTING_ROI_RECT[0], COUNTING_ROI_RECT[1]), (COUNTING_ROI_RECT[2], COUNTING_ROI_RECT[3]), (0, 0, 255), 1)

    if cached_boxes is not None:
        for box in cached_boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if center_x < CABIN_ZONE_X_END:
                    if (COUNTING_ROI_RECT[0] < center_x < COUNTING_ROI_RECT[2]) and \
                       (COUNTING_ROI_RECT[1] < center_y < COUNTING_ROI_RECT[3]):
                        
                        bbox_area = (x2 - x1) * (y2 - y1)
                        person_value = 1
                        color = (0, 255, 0)

                        if bbox_area > LARGE_PERSON_AREA_THRESHOLD:
                            person_value = 2
                            color = (0, 0, 255)
                        
                        current_frame_count += person_value

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Val:{person_value}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif center_x > WAITING_ZONE_X_START:
                    waiting_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if current_frame_count >= MAX_CAPACITY:
        raw_status = "Full"
    elif current_frame_count == 0:
        raw_status = "Empty"
    else:
        raw_status = "Occupied"

    status_buffer.append(raw_status)

    if len(status_buffer) < SEQUENCE_LENGTH:
        stable_status = raw_status
        stability_score = int((len(status_buffer)/SEQUENCE_LENGTH) * 100)
    else:
        most_common = Counter(status_buffer).most_common(1)
        stable_status = most_common[0][0]
        count_occurrence = most_common[0][1]
        stability_score = int((count_occurrence / SEQUENCE_LENGTH) * 100)

    if stable_status == "Full":
        door_permission = "DO NOT OPEN"
        display_color = (0, 0, 255)
    elif stable_status == "Empty":
        door_permission = "OK TO OPEN"
        display_color = (0, 255, 0)
    else:
        door_permission = "OK TO OPEN"
        display_color = (0, 255, 255)

    display_panel = np.zeros((160, w, 3), dtype=np.uint8)
    
    cv2.putText(display_panel, "--- AI DECISION ---", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_panel, f"Real-time Count: {current_frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(display_panel, f"AI Stable Status: {stable_status} (Conf: {stability_score}%)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
    cv2.putText(display_panel, f"Door Cmd: {door_permission}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(display_panel, "--- WAITING AREA ---", (WAITING_ZONE_X_START + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_panel, f"People Waiting: {waiting_count}", (WAITING_ZONE_X_START + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, stat in enumerate(status_buffer):
        c = (100, 100, 100)
        if stat == "Full": c = (0, 0, 255)
        elif stat == "Empty": c = (0, 255, 0)
        elif stat == "Occupied": c = (0, 255, 255)
        
        cx = 350 + (i * 20)
        cv2.circle(display_panel, (cx, 30), 6, c, -1)
        cv2.circle(display_panel, (cx, 30), 7, (255,255,255), 1)

    combined_frame = np.vstack((frame, display_panel))

    cv2.imshow("Elevator Watch System", combined_frame)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
