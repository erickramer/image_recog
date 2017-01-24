from flask import render_template, request, jsonify
from image_recog import app
from binascii import a2b_base64
import PIL.Image
import StringIO
import ml
import scipy.misc
import io
import numpy as np

score = ml.ImageScorer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/score", methods=["POST"])
def api_score():
    file_type, data = request.form['image'].split(',')
    bindata = data.decode("base64")
    mean_pixel = [103.939, 116.779, 123.68]

    f = io.BytesIO(bindata)

    s = score(f)

    return jsonify(s)

@app.route("/static/<path:path>")
def static_path(path):
    return app.send_static_file(path)
