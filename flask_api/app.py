from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import torch

model = torch.load("../game_hands.pt")

def predict_image(filepath):
   
    predictions = model.predict(filepath)
    
    return predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = "supersecretkey"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'predict_pic.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predict_pic.png')
            file.save(filepath)

            # Call your model's prediction function
            prediction = predict_image(filepath)

            return render_template('result.html', filename=filename, prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


