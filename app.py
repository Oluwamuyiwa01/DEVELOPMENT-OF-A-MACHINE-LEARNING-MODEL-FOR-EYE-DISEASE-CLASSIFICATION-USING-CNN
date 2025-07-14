import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)


model = load_model("Model/Eye_disease_model.h5")


class_names = ["Cataract", "Diabetic Retinopathy", "Glaucoma"]


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  
            img = np.expand_dims(img, axis=0)  

            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100  

            
            prediction_text = f"Predicted: {class_names[predicted_class]} ({confidence:.2f}%)"
            return render_template("index.html", prediction=prediction_text, image_path=file_path)

    return render_template("index.html", prediction=None, image_path=None)


if __name__ == "__main__":
    app.run(debug=True)





