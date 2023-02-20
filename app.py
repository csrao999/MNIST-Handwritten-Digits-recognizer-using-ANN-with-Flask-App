from unittest import result
import cv2
import os
from PIL import Image
from numpy import asarray
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model

model = load_model('digit_recognizer.h5')

app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader" , methods=['GET', 'POST'])
def uploader():    
    if request.method=='POST':
        f = request.files['file1']
        f.filename = "image.jpg"
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        img = Image.open("static/image.jpg")
        img = asarray(img)
        size = (28,28) 
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = model.predict(img.reshape(1,28,28)).argmax(axis=1)[0]
        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
        return render_template("uploaded.html", predicted_digit=result, input_image=pic1)

if __name__ == '__main__':
    app.run(debug=True) 