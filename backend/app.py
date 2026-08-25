from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import os
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(name):
    ensure_attendance_file()
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    today = datetime.now().strftime("%Y-%m-%d")
    if not df.empty and "Name" in df.columns and "Date" in df.columns:
        if ((df["Name"] == name) & (df["Date"] == today)).any():
            return

    now = datetime.now()
    new_row = pd.DataFrame([{
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S")
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/api/register", methods=["POST"])
def register_student():
    name = request.form.get("name", "").strip()
    image = request.files.get("image")

    if not name or not image:
        return jsonify({"error": "Name and image are required"}), 400

    person_dir = os.path.join(IMAGES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    path = os.path.join(person_dir, filename)
    image.save(path)

    return jsonify({"message": f"Image saved for {name}. Now run encode_faces.py"}), 201

@app.route("/api/capture-snapshot", methods=["POST"])
def capture_snapshot():
    # Opens camera briefly to grab one current frame for snapshot storage
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap.release()

    if not success:
        return jsonify({"error": "Failed to access camera for snapshot"}), 500

    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame)

    return jsonify({"message": f"Snapshot saved successfully as {filename}!"})

def generate_frames():
    known_encodings, known_names = load_encodings()
    print("DEBUG: Loaded names ->", known_names)
    print("DEBUG: Total encodings loaded ->", len(known_encodings))

    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            # Resize to 1/4 size for fast face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = []
            face_names = []

            if len(known_encodings) > 0:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    name = "Unknown"

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            mark_attendance(name)

                    face_names.append(name)

            # Draw boxes and names on the full-size frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        video_capture.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/attendance")
def get_attendance():
    ensure_attendance_file()
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/rebuild-encodings", methods=["POST"])
def rebuild_encodings():
    os.system("python encode_faces.py")
    return jsonify({"message": "Encodings rebuilt successfully"})

if __name__ == "__main__":
    app.run(debug=True)
