# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:21:42 2018

@author: Camila
"""
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
#from scipy.misc import imsave, imread, imresize

import sys

import os

sys.path.append(os.path.abspath("./model"))
#import load

#initialize our flask app
app = Flask(__name__)

photos = UploadSet('photos',IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app,photos)

@app.route('/',method=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        rec = Photo(filename=filename, user=g.user.id)
        rec.store()
        flash("Photo saved.")
        return filename
    return render_template('uploads.html')

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 7000))
    app.run(host='0.0.0.0', port=port)
 #   app.run(debug=True)

    
