from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
import cv2
import uuid

app = Flask(__name__)

# Load model sekali di awal
model = YOLO("yolov8/best.pt")

# Folder simpan gambar sementara
UPLOAD_FOLDER = "temp_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Selada Detection API is running 🌱"

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, f"{img_id}_detected.jpg")

    # Simpan gambar yang dikirim
    file.save(input_path)

    # Deteksi pake YOLO
    results = model(input_path)

    # Simpan hasil dengan bounding box
    annotated_frame = results[0].plot()
    cv2.imwrite(output_path, annotated_frame)

    # Balikin file hasil deteksi ke user
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
