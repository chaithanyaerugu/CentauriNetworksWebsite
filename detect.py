import csv
import os
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
CSV_FILE = os.path.join(BASE_DIR, "audience_log.csv")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Windows-friendly webcam initialization. CAP_DSHOW often avoids camera-open issues.
def open_camera():
    for index in (0, 1, 2):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Camera opened successfully: index {index}")
            return cam
        cam.release()
    raise RuntimeError(
        "Could not open a webcam. Close Windows Camera, Teams, Zoom, OBS or other apps using it, "
        "then try again. You can also change CAMERA_INDEX below."
    )

CAMERA_INDEX = 0
camera = open_camera()

lock = threading.Lock()
latest_jpeg = None
metrics = {
    "people": 0,
    "engaged": 0,
    "attention_rate": 0.0,
    "avg_dwell": 0.0,
    "peak_audience": 0,
    "qr_interactions": 0,
    "updated": ""
}

tracks = {}
peak_audience = 0
last_csv_save = 0.0
qr_interactions = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
qr_detector = cv2.QRCodeDetector()


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def estimate_facing_screen(person_box, gray):
    """Prototype attention estimate; not biometric gaze detection."""
    x1, y1, x2, y2 = person_box
    h_img, w_img = gray.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    w, h = x2 - x1, y2 - y1
    if w < 35 or h < 70:
        return False

    upper = gray[y1:max(y1 + int(h * 0.65), y1 + 1), x1:x2]
    if upper.size == 0:
        return False

    faces = face_detector.detectMultiScale(
        upper, scaleFactor=1.12, minNeighbors=5, minSize=(24, 24)
    )
    for fx, fy, fw, fh in faces:
        cx = fx + fw / 2
        relative_x = cx / max(1, w)
        if 0.25 <= relative_x <= 0.75 and fw >= max(24, int(w * 0.12)):
            return True
    return False


def update_track_state(track_id, box, engaged_now, now):
    key = track_id if track_id >= 0 else f"unknown_{id(box)}"
    state = tracks.setdefault(
        key, {"first_seen": now, "last_seen": now, "engaged_time": 0.0}
    )
    dt = min(1.0, max(0.0, now - state["last_seen"]))
    if engaged_now:
        state["engaged_time"] += dt
    state["last_seen"] = now
    state["box"] = box
    state["engaged"] = engaged_now
    return state


def process_frame(frame):
    global peak_audience, last_csv_save, qr_interactions

    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model.track(
        frame,
        persist=True,
        classes=[0, 67],  # COCO: person, cell phone
        conf=0.35,
        iou=0.50,
        verbose=False
    )

    people = []
    phones = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # IMPORTANT: each `box` returned by Ultralytics contains exactly one
        # detection. Therefore class/coordinate tensors must be indexed at 0
        # inside the individual box, not by the outer enumeration index.
        ids = boxes.id
        for idx in range(len(boxes)):
            box = boxes[idx]
            cls = int(box.cls[0].item())
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            track_id = int(ids[idx].item()) if ids is not None and idx < len(ids) else -1

            if cls == 0:
                people.append((track_id, xyxy))
            elif cls == 67:
                phones.append(xyxy)

    current_count = len(people)
    peak_audience = max(peak_audience, current_count)
    engaged_count = 0
    dwell_values = []

    for track_id, box in people:
        engaged_now = estimate_facing_screen(box, gray)
        phone_near_person = any(iou(box, phone) > 0.05 for phone in phones)
        state = update_track_state(track_id, box, engaged_now, now)
        if engaged_now:
            engaged_count += 1
        dwell_values.append(now - state["first_seen"])

        label = f"ID {track_id if track_id >= 0 else '?'}"
        if engaged_now:
            label += "  LIKELY LOOKING"
        if phone_near_person:
            label += "  PHONE"

        x1, y1, x2, y2 = box
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 220, 120) if engaged_now else (120, 120, 120), 2
        )
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (0, 255, 255) if engaged_now else (210, 210, 210), 2
        )

    attention_rate = (engaged_count / current_count * 100.0) if current_count else 0.0
    avg_dwell = sum(dwell_values) / len(dwell_values) if dwell_values else 0.0

    # Detect whether the billboard QR is visible. This does not prove a scan.
    qr_data, qr_points, _ = qr_detector.detectAndDecode(frame)
    if qr_points is not None:
        pts = qr_points.astype(int).reshape(-1, 2)
        cv2.polylines(frame, [pts], True, (255, 0, 255), 3)
        cv2.putText(frame, "QR VISIBLE", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    if now - last_csv_save >= 30:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    "Time", "AudienceCount", "Engaged", "AttentionRate", "AvgDwell", "PeakAudience", "QRInteractions"
                ])
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_count,
                engaged_count, round(attention_rate, 2), round(avg_dwell, 2),
                peak_audience, qr_interactions
            ])
        last_csv_save = now

    cv2.rectangle(frame, (10, 10), (410, 150), (10, 10, 10), -1)
    cv2.putText(frame, f"People: {current_count}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Likely engaged: {engaged_count}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
    cv2.putText(frame, f"Attention: {attention_rate:.0f}%", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, f"QR interactions: {qr_interactions}", (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    with lock:
        metrics.update({
            "people": current_count,
            "engaged": engaged_count,
            "attention_rate": round(attention_rate, 1),
            "avg_dwell": round(avg_dwell, 1),
            "peak_audience": peak_audience,
            "qr_interactions": qr_interactions,
            "updated": datetime.now().strftime("%H:%M:%S")
        })

    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if ok else None


def camera_loop():
    global latest_jpeg
    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue
        try:
            jpeg = process_frame(frame)
            if jpeg:
                with lock:
                    latest_jpeg = jpeg
        except Exception as exc:
            # Keep the camera thread alive so one bad frame cannot freeze the
            # dashboard. The full traceback is printed for debugging.
            import traceback
            print("Camera processing error:", exc)
            traceback.print_exc()
            time.sleep(0.2)


@app.route("/")
def index():
    return render_template("marketing.html")

@app.route("/network")
def network():
    return render_template("network.html")

@app.route("/ai")
def ai_dashboard():
    return render_template("ai.html")


@app.route("/count")
def count():
    with lock:
        return jsonify(metrics)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                frame = latest_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/qr_scan", methods=["GET", "POST"])
def qr_scan():
    global qr_interactions
    with lock:
        qr_interactions += 1
        count_value = qr_interactions
    return jsonify({"ok": True, "qr_interactions": count_value})


if __name__ == "__main__":
    print("\nCENTAURI NETWORKS — AI ANALYTICS")
    print("Dashboard: http://127.0.0.1:5000")
    print("Do NOT use VS Code Live Server (port 5500).\n")
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
