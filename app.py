from flask import Flask
from flask import render_template,flash,redirect,request,url_for
from forms import LoginForm, DataForm,TestForm
from config import Config
import subprocess
import sys
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
from upload import upload_for_tensorboard,run_model

UPLOAD_FOLDER = '/home/datasets'
MODEL_FOLDER = '/home/ayushmaan/trained_model'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/')
@app.route('/index')
def index():
    user = {'username':'user'}
    return render_template('index.html',user=user)


@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        #flash('usernameis:{},remember it:{}'.format(
            #form.username.data,form.remember_me.data))
        #v = sys.version
        #flash(v)
        return redirect('/index')
    return render_template('login.html',title='Log in',form=form)


@app.route('/evaluator',methods=['GET','POST'])
def evaluator():
    form= DataForm()
    if form.validate_on_submit():
        data1=form.data1.data
        data2=form.data2.data
        # evaluate here
        return redirect('/result')
    return render_template('evaluator.html',title='Evaluator',form=form)

@app.route('/test',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #temp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    return render_template('test.html')

@app.route('/uploadForm',methods=['GET','POST'])
def uploadFile():
    form = TestForm()
    if request.method == 'POST':
        labels = form.Labels.data
        target = form.Target.data
        filename = secure_filename(form.file.data.filename)
        form.file.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',filename=filename))
    return render_template('upload.html',title='Form Uploader',form=form)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    file_path = run_model(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    model_path = MODEL_FOLDER
    what_if_path = upload_for_tensorboard(file_path,model_path)
    #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    return redirect(what_if_path)



if __name__ == '__main__':
    app.run()
