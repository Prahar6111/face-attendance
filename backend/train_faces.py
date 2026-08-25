import os

base = "data/images"
os.makedirs(base, exist_ok=True)
print("Create folders inside data/images/<student_name> and place images there.")
print("Then run: python encode_faces.py")