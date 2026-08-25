import os
import pickle
import face_recognition

DATA_DIR = "data/images"
ENCODINGS_FILE = "data/encodings.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"Saved {len(known_names)} face encodings to {ENCODINGS_FILE}")