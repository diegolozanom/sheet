import tensorflow as tf
from flask import Flask, request
from PIL import Image
import io
import os
import piexif
import base64
import math
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello():
     """Return a friendly HTTP greeting."""
     return str("FUNCIONA")


@app.route('/api/imagen', methods=['GET', 'POST'])
def imagen():
    imagefile = None
    imagefile = request.files['image']
    if imagefile:
       img = Image.open(file)
    
    return "Image successfully loaded."


@app.route('/api/test', methods=['POST'])
def post():
    r = request
    result = sheet_return(r.data)
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "modelo_final"))
    predictions = model.predict(result)
    value = int(np.mean(np.array([np.argmax(predict) for predict in predictions])))
    return value


def rotate_jpeg(filename):
    img = filename
    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])

        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            exif_bytes = piexif.dump(exif_dict)

            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180)
            elif orientation == 4:
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

    return img


def preprocess(img):
    img_tf = tf.keras.preprocessing.image.img_to_array(img) / 255.
    img_tf = tf.image.resize(img_tf, (224, 224), method='lanczos5', antialias=True)
    img_tf = np.array(img_tf, dtype=np.float32)

    return img_tf


def sheet_return(img):
    n = 9
    columns = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / float(columns)))
    image_64_decode = base64.decodebytes(img)
    image_open = Image.open(io.BytesIO(image_64_decode))
    image_rotated = rotate_jpeg(image_open)
    im_w, im_h = image_rotated.size
    tile_w, tile_h = int(math.floor(im_w / columns)), int(math.floor(im_h / rows))
    tiles = []
    for pos_y in range(0, im_h - rows, tile_h):
        for pos_x in range(0, im_w - columns, tile_w):
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = image_rotated.crop(area)
            tiles.append(preprocess(image))

    tiles = np.stack(tiles, axis=0)

    return tiles


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
