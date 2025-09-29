import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


app = Flask(__name__)


MODEL_PATH = 'vietnamese_currency_model_cnn.h5'
CLASS_NAMES_PATH = 'vietnamese_currency_class_names.npy'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


try:
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    print(f"[*] Model '{MODEL_PATH}' và class names đã được tải thành công.")
    print(f"[*] Các lớp được nhận dạng: {class_names}")
except Exception as e:
    print(f"[!] Lỗi khi tải model hoặc class names: {e}")
    model = None
    class_names = []


def preprocess_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
       
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None

        #
        resized_img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        
        normalized_img = resized_img.astype('float32') / 255.0
        
        
        input_img = normalized_img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        
        return input_img
    except Exception as e:
        print(f"[!] Lỗi trong quá trình xử lý ảnh: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes)
            
            if processed_image is None:
                return jsonify({'error': 'Could not process the image.'}), 400

            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            
            print(f"[*] Dự đoán: {predicted_class_name}")

            return jsonify({'prediction': str(predicted_class_name)})

        except Exception as e:
            print(f"[!] Lỗi khi dự đoán: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({'error': 'Unknown error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)