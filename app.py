from flask import Flask, render_template, request, url_for, session
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
import cv2
import tensorflow as tf
import numpy as np
# Flask configuration and initial setup
app = Flask(__name__)
app.secret_key = 'celebAI'
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    # No image to display initially
    return render_template('home.html', image_url=None)
from tensorflow.keras.preprocessing.image import img_to_array

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    counter = 1
    while os.path.exists(filepath):
        name, extension = os.path.splitext(filename)
        filename = f"{name}_{counter}{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        counter += 1
    file.save(filepath)

    img = cv2.imread(filepath)  # Load the image in color
    if img is None:
        return "Image not loaded properly"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=10, minSize=(64, 64), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        img = cv2.resize(img, (64, 64))
    else:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (64, 64))

    img = np.expand_dims(img, axis=0)  # Add the batch dimension

    # Ensure the image has 3 channels (important if the original image is grayscale)
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    model = tf.keras.models.load_model("celebrity_recognition_model2.h5")
    predictions = model.predict(img)
    predicted_classes = np.argmax(predictions, axis=1)
    classes = ["Aaron Judge", "Aaron Paul", "Aaron Taylor-Johnson", "Abigail Breslin", "Adam Sandler", "Adele", "Adriana Barraza", "Adriana Lima", "Adrianne Palicki", "Adrien Brody", "Akemi Darenogare", "Al Pacino", "Al Roker", "Alan Alda", "Alan Arkin", "Alan Rickman", "Albert Brooks", "Albert Finney", "Alec Baldwin", "Alessandra Ambrosio", "Alex Pettyfer", "Alexander Skarsgard", "Alexandra Daddario", "Alexis Thorpe", "Ali Larter", "Alice Eve", "Alicia Vikander", "Alx James", "Amanda Bynes", "Amanda Crew", "Amanda Peet", "Amanda Seyfried", "Amber Heard", "Amy Adams", "Amy Ryan", "Amy Schumer", "Analeigh Tipton", "Anderson Cooper", "Andie MacDowell", "Andreea Diaconu", "Andrew Garfield", "Andrew Lincoln", "Andrew Luck", "Andy Garcia", "Andy Murray", "Andy Samberg", "Andy Serkis", "Angela Bassett", "Angelina Jolie", "Anjelica Huston", "Anna Faris", "Anna Friel", "Anna Kendrick", "Anna Paquin", "Anna Sui", "AnnaSophia Robb", "Anne Bancroft", "Anne Baxter", "Anne Hathaway", "Annette Bening", "Anthony Hopkins", "Anthony Mackie", "Anthony Perkins", "Antonio Banderas", "Armin Mueller-Stahl", "Arnold Schwarzenegger", "Art Carney", "Ashley Graham", "Ashley Greene", "Ashley Judd", "Ashton Kutcher", "Audrey Hepburn", "Audrey Tautou", "Ava Gardner", "Barabara Palvin", "Barbra Streisand", "Barry Pepper", "Bella Hadid", "Ben Affleck", "Ben Foster", "Ben Johnson", "Ben Kingsley", "Ben Stiller", "Benedict Cumberbatch", "Benicio Del Toro", "Benjamin Bratt", "Berenice Bejo", "Bernie Mac", "Bette Midler", "Betty White", "Beyonce Knowles", "Bill Daley", "Bill Hader", "Bill Murray", "Bill O Reilly", "Bill Paxton", "Bill Pullman", "Bill Rancic", "Billy Bob Thornton", "bella thorne"]
    final_pred_class = classes[predicted_classes[0]]


    return render_template('prediction.html', pred=final_pred_class)

if __name__ == '__main__':
    app.run(port=4000, debug=True)
