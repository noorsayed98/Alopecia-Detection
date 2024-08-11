from flask import Flask, render_template, request, redirect, url_for
# import cv2 as cv
import numpy as np
# import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
# from PIL import Image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

# Function to perform prediction
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.
    img_array = np.expand_dims(img_array, axis=0)

    # Load models
    resnet50_model = load_model('/home/bizzle/Documents/Responsive-Homepage-2-main/models/resnet50_model.h5')
    svm_cnn_model = load_model('/home/bizzle/Documents/Responsive-Homepage-2-main/models/svm_using_cnn_smote.h5')
    vgg19_model = load_model('/home/bizzle/Documents/Responsive-Homepage-2-main/models/vgg19_SMOTE_ksh.h5')

    # Make predictions
    resnet50_prediction = resnet50_model.predict(img_array)
    svm_cnn_prediction = svm_cnn_model.predict(img_array)
    vgg19_prediction = vgg19_model.predict(img_array)

    # Interpret predictions
    resnet50_result = 'Alopecia' if resnet50_prediction < 0.5 else 'Healthy Hair'
    svm_cnn_result = 'Alopecia' if svm_cnn_prediction < 0.5 else 'Healthy Hair'
    vgg19_result = 'Alopecia' if vgg19_prediction < 0.5 else 'Healthy Hair'

    # Ensembling
    ensemble_prediction = [resnet50_result, svm_cnn_result, vgg19_result]
    final_prediction = 'Alopecia Detected' if ensemble_prediction.count('Alopecia') >= 2 else 'Healthy Hair'

    return final_prediction

@app.route('/Imageupload.html', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        img = request.files['Scalp']
        img.save('uploaded_image.jpg')
        return redirect(url_for('result'))
    return render_template('Imageupload.html')

@app.route('/result')
def result():
    prediction = predict('uploaded_image.jpg')
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)