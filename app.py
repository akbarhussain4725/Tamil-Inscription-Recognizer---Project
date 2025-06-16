from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from tensorflow import keras
from googletrans import Translator
from gtts import gTTS
import uuid
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths and setup
IMG_SIZE = 64
MODEL_PATH = r"C:\Users\hussa\OneDrive\Desktop\PROJECT\tamil_cnn_model.keras"
FOLDER_A = r"C:\Users\hussa\OneDrive\Desktop\PROJECT\folderA"
app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH)
class_labels = sorted(os.listdir(FOLDER_A))
reverse_label_map = {i: label for i, label in enumerate(class_labels)}
translator = Translator()

# Preprocessing function (from your Colab code)
def preprocess_inscription_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

# Character segmentation and recognition
def recognize_characters(image_bytes):
    thresh = preprocess_inscription_image(image_bytes)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sorted([cv2.boundingRect(cnt) for cnt in contours], key=lambda b: b[0])

    recognized_chars = []
    for x, y, w, h in bounding_boxes:
        if w < 5 or h < 5:
            continue
        roi = thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi, verbose=0)
        index = np.argmax(pred)
        char = reverse_label_map.get(index, '?')
        recognized_chars.append((x, char))

    recognized_chars.sort(key=lambda item: item[0])
    return ''.join([char for _, char in recognized_chars])

# Translation function
def translate_to_english(tamil_text):
    if tamil_text == "காசி":
        return "kasi"
    try:
        translated = translator.translate(tamil_text, src="ta", dest="en").text
        return translated
    except Exception as e:
        return "Translation Error"

# Text-to-speech function
def generate_tts(text):
    filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    tamil_text = recognize_characters(image.read())
    english_text = translate_to_english(tamil_text)
    audio_path = generate_tts(english_text)

    return jsonify({
        "tamil_text": tamil_text,
        "english_text": english_text,
        "audio": audio_path
    })

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)