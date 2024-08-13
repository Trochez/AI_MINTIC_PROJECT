from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
from model import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Read image file in memory
            img_bytes = file.read()
            img_stream = BytesIO(img_bytes)

            # Call the prediction function from model.py
            predicted_class = predict_image(img_stream)

            return jsonify({'prediction': predicted_class})
    
    except OSError as e:
        return jsonify({'error': 'An error occurred while processing the image. Please try again with a different image or smaller size.'})

if __name__ == '__main__':
    app.run(debug=True)
