# Image Recognition as a WebApp


## Install

1) Clone the repo

```
git clone https://github.com/erickramer/image_recog.git
cd image_recog
```

2) Download VGG weights from [here](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view) and put the weights in the data folder

3) Create virtualenv and install requirements.txt

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

4) Make sure your ~/.keras/keras.json looks like this:

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```
5) Run app

```
python run_app.py
```

6) Browse to [localhost:5000](localhost:5000)
