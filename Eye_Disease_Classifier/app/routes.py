import os
from ultralytics import YOLO
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
import numpy as np

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Model inference logic here
            result = run_inference(filepath)

            return render_template('index.html', filepath=filepath, filename=filename, result=result)

    return render_template('index.html')

def run_inference(image_path):
    model_path = r'model\best.pt'
    model = YOLO(model_path)
    results = model(image_path)
    disease_classes = {0: 'cataract', 1: 'diabetic_retinopathy', 2: 'glaucoma', 3: 'normal'}
    disease_pred = disease_classes[results[0].probs.top5[0]]
    disease_conf = results[0].probs.top5conf.tolist()[0]
    disease_result = disease_pred + " Prob: "+ str(round(disease_conf,4))
    return disease_result
