from flask import render_template, request, jsonify
from image_recog import app, img_count
from binascii import a2b_base64
import PIL.Image
import StringIO
import ml
import scipy.misc
import io
import numpy as np

model = ml.VGG_16()
tags = open("./data/tags.txt", "r").readlines()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/score", methods=["POST"])
def score():
    file_type, data = request.form['image'].split(',')
    bindata = data.decode("base64")
    mean_pixel = [103.939, 116.779, 123.68]

    f = io.BytesIO(bindata)

    img = scipy.misc.imread(f)
    print img.shape
    img = img[:, :, 0:3]
    img = scipy.misc.imresize(img, (224, 224))
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)
    ind = np.argmax(p)

    return jsonify(tags[ind])

@app.route("/static/<path:path>")
def static_path(path):
    return app.send_static_file(path)
