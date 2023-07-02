from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
model = load_model('./models/face_auth_99.5p_dropout.h5')

app = Flask(__name__)

DESIRED_SIZE = 1024

def crop_image(image):
    #Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(DESIRED_SIZE//2, DESIRED_SIZE//2))
    if len(faces) == 0: assert ValueError("Picture doesnt contain any faces")
    (x, y, w, h) = faces[0]
    center_x = x + (w // 2)
    center_y = y + (h // 2)
    crop_x = center_x - DESIRED_SIZE//2
    crop_y = center_y - DESIRED_SIZE//2
    return image[crop_y:crop_y+DESIRED_SIZE, crop_x:crop_x+DESIRED_SIZE]

def prep_image(image_path):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    print(img.shape)
    if h == 1024 and w == 1024 and c == 3:
        return img/255.0
    if h < 1024 or w < 1024:
        raise ValueError("Image size too small")
    if c != 3:
        channels = cv2.split(img)
        img = cv2.merge([channels[2], channels[1], channels[0]])
    if h > 1024 or w > 1024: img = crop_image(img)
    cv2.imwrite(os.path.join('static', 'temp_image.png'), img)
    return img/255.0
    


def predict_image(img):
    return "Real" if model.predict(np.array([img]))[0][0] > 0.5 else "Fake"

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method != 'POST':
        return render_template('index.html')
    if 'temp_image.png' in os.listdir('./static'):
        os.remove('./static/temp_image.png')
    if 'image' not in request.files:
        return render_template('index.html', error='No image file selected')
    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error='No image file selected')
    image_path = os.path.join('static', 'temp_image.png')
    image_file.save(image_path)
    try:
        img = prep_image(image_path)
        result = predict_image(img)
    except Exception as err:
        return render_template('index.html', error='err')
    return render_template('index.html', result=result, image_filename='temp_image.png')

if __name__ == '__main__':
    app.run(debug=True)