import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template

app = Flask(__name__)

model = load_model("logo_detection_model.h5", compile=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            if not os.path.exists(os.path.join(basepath, 'uploads')):
                os.makedirs(os.path.join(basepath, 'uploads'))
            f.save(filepath)

            img = image.load_img(filepath, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            y = model.predict(x)
            preds = np.argmax(y, axis=1)
            index = ['Adidas', 'Amazon', 'Android', 'Apple', 'Ariel', 'BMW', 'Bic', 'Burger King', 'Cadbury',
                     'Chevrolet', 'Chrome', 'Coca Cola', 'Cowbell', 'Dominos', 'Fila', 'Gillette', 'Google',
                     'Goya oil', 'Guinness', 'Heinz', 'Honda', 'Hp', 'Huawei', 'Instagram', 'Kfc', 'Krisspy Kreme',
                     'Lays', 'Levis', 'Lg', 'Lipton', 'Mars', 'Marvel', 'McDonald', 'Mercedes Benz', 'Microsoft',
                     'MnM', 'Mtn', 'Mtn dew', 'NASA', 'Nescafe', 'Nestle', 'Nestle milo', 'Netflix', 'Nike',
                     'Nutella', 'Oral b', 'Oreo', 'Pay pal', 'Peak milk', 'Pepsi', 'PlayStation', 'Pringles',
                     'Puma', 'Reebok', 'Rolex', 'Samsung', 'Sprite', 'Starbucks', 'Tesla', 'Tiktok', 'Twitter',
                     'YouTube', 'Zara']
            if preds[0] <= len(index):
                text = "The given logo is fake"
            else:
                text = "The given logo is of "+str(index[preds[0]])
            return text
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=False)
