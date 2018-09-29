from flask import render_template, request, jsonify
from image_recog import app
from binascii import a2b_base64
import PIL.Image
import StringIO
import ml
import scipy.misc
import io
import numpy as np

img_scorer = ml.ImageScorer(logger=app.logger)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/score", methods=["POST"])
def api_score():

    # parse image from webcam
    file_type, data = request.form['image'].split(',')
    bindata = data.decode("base64")
    f = io.BytesIO(bindata)
    
    # read model requested
    model = request.form['model']

    s = img_scorer.score(f, model=model)

    return jsonify(s)

@app.route("/static/<path:path>")
def static_path(path):
    return app.send_static_file(path)
