# 🖋️ Tamil Handwritten Character Recognition & Translation

A deep learning-based web application that recognizes **Tamil handwritten characters** from images, translates the recognized text to **English**,
and provides **text-to-speech (TTS)** output for both Tamil and English.



## 🔍 Features

- ✅ **Handwritten Tamil Character Recognition** using a CNN model
- ✅ Supports all **247 Tamil characters** (Uyir, Mei, Uyirmei, Grantha, Special)
- ✅ **Word segmentation** and prediction from line images
- ✅ **English Translation** using Google Translate API
- ✅ **Text-to-Speech (TTS)** for both Tamil and English using `gTTS`
- ✅ **Web Interface** for uploading images and viewing results
- ✅ **Augmented Dataset** with rotation, blur, noise, brightness, etc.



## 🧠 Model Info

- **Framework**: TensorFlow / Keras
- **Architecture**: CNN (Convolutional Neural Network)
- **Input Size**: 128x128 grayscale images
- **Training Data**: ~247,000 augmented samples
- **Accuracy**: > 95% on test data



## How to Run Locally
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `python app.py`






## Project Structure
- `app.py` - Flask server
- `tamil_cnn_model.keras` - Trained CNN model
- `templates/index.html` - Frontend UI
- `utils/` - Helper scripts for segmentation, translation, TTS


