from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

11# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

14# Load your trained CNN model
model = load_model('model/model.h5')
16
17# Load class labels for Plant Village dataset
labels = [
    'Apple Scab',
    'Apple Black Rot',
    'Apple Cedar Rust',
    'Healthy Apple',
    'Blueberry Healthy',
    'Cherry Powdery Mildew',
    'Cherry Healthy',
    'Corn Gray Leaf Spot',
    'Corn Healthy',
    'Grape Black Rot',
    'Grape Esca',
    'Grape Leaf Blight',
    'Grape Healthy',
    'Orange Haunglongbing',
    'Peach Bacterial Spot',
    'Peach Healthy',
    'Pepper Bell Bacterial Spot',
    'Pepper Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight',
    'Potato Healthy',
    'Raspberry Healthy',
    'Soybean Healthy',
    'Strawberry Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Healthy'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))  # Adjust to model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_class = labels[predicted_index]

        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)