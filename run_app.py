from image_recog import app

if __name__ == "__main__":
    app.config['TESTING'] = True
    app.config['DEBUG'] = True
    app.config['ENV'] = 'dev'
    app.run()
