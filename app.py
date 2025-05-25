from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Resize to model's expected input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 128, 128, 3)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded.", image_url=None)
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.", image_url=None)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    input_data = preprocess_image(filepath)
    prediction = model.predict(input_data)
    predicted_class = "Dog" if prediction[0] > 0.5 else "Cat"
    return render_template('index.html', prediction=predicted_class, image_url=url_for('static', filename='uploads/' + file.filename))

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


# app.py
# This is a simple Flask web application that serves a machine learning model.


