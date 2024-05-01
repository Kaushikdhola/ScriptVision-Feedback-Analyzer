import base64
import os
import re
import pickle
from glob import glob
import io
import cv2
from flask_mysqldb import MySQL
from flask_dropzone import Dropzone
import pytesseract
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask import Flask, render_template, request, session, jsonify, send_file, flash, url_for, redirect
import requests
import time
import MySQLdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from docutils.nodes import classifier
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY'] = 'YOUR_KEY'
os.environ['COMPUTER_VISION_ENDPOINT'] = 'YOUR_ENDPOINT'


records = []

if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print(
        "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to "
        "take effect.**")
    sys.exit()

if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']

text_recognition_url = endpoint + "vision/v2.1/read/core/asyncBatchAnalyze"

app = Flask(__name__, template_folder='Templates/',static_url_path='/static')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

labelList = []
# secret key for session
app.config['SECRET_KEY'] = ''

# Dropzone  Implementation

dropzone = Dropzone(app)
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'index'
app.config[
    'DROPZONE_DEFAULT_MESSAGE'] = "<img src='../static/cloud-upload-512.png' style='width:200px; height:200px; margin-left:5%; margin-top:10%;'>"
plot_url=''
# database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'hack'

mysql = MySQL(app)

# Location for storing images
app.config['UPLOADED_PHOTOS_DEST'] = "\Images"
app.config['UPLOAD_DOWNLOAD'] = "\Images"
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

UPLOAD_FOLDER = ""
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

patterns = []
d1 = {}


def Mytest(df1):
   
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    with open('tfidfCV.pickle', 'rb') as f:
        tfidf = pickle.load(f)

    dic = {'Feedback': [], 'Prediction': []}
    dic_positive = {'Prediction': []}
    dic_negative = {'Prediction': []}

    column1 = df1.columns

    for column in column1:
        dic_positive[column] = []
        dic_negative[column] = []

    for i in range(len(df1.index)):

        review = df1['Feedback'][i].lower()
        review = review.split()

        lemmatizer = WordNetLemmatizer()

        review = [lemmatizer.lemmatize(word) for word in review
                  if not word in set(stopwords.words('english'))]

        review = ' '.join(review)
        li = [review]
        res = tfidf.transform(li)
        res = model.predict(res)
        res = res[0]

        if (res == 1):
            dic.setdefault('Feedback', []).append(df1['Feedback'][i])
            dic.setdefault('Prediction', []).append('Positive')
            dic_positive.setdefault('Prediction', []).append('Positive')
            for column in column1:
                dic_positive.setdefault(column, []).append(df1[column][i])
        else:
            dic.setdefault('Feedback', []).append(df1['Feedback'][i])
            dic.setdefault('Prediction', []).append('Negative')
            dic_negative.setdefault('Prediction', []).append('Negative')
            for column in column1:
                dic_negative.setdefault(column, []).append(df1[column][i])

    df2 = pd.DataFrame(dic)

    df3 = pd.DataFrame(dic_positive)
    df3.to_excel(r'positive_prediction.xlsx', index=None, header=True)

    df4 = pd.DataFrame(dic_negative)
    df4.to_excel(r'negative_prediction.xlsx', index=None, header=True)

    output = io.BytesIO()
    plt.hist(df2['Prediction'])
    plt.savefig(output,format='png')
    output =io.BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)
    global plot_url
    plot_url = base64.b64encode(output.getvalue()).decode()
    render_template("prediction.html",url=plot_url)

def replace_all(text, patterns):
    symbol = [':', '-', '=', '~']
    for i in patterns:
        if "".join(i.lower().split(' ')) in ("".join(text.lower().split(' '))):
            text = text.lower().replace(i.lower(), '').strip()
    for j in symbol:
        text = text.replace(j, '').strip()
    return text


def single_pattern(record, line, pattern):
    value = replace_all(line, patterns)
    for i in record[record.index(line) + 1:]:
        flag = True
        for j in patterns:
            if "".join(j.lower().split(' ')) not in ("".join(i.split(" "))).lower():
                flag = True
            else:
                flag = False
                break
        if flag:
            value += ' ' + i.strip().lower()
        else:
            break
    if "".join(pattern.lower().split(' ')) in ["mobilenumber", "email"]:
        d1.setdefault(pattern.title(), []).append(("".join(value.split(" "))).lower())
    else:
        d1.setdefault(pattern.title(), []).append(value)

def data_cleaning(records):
    for record in records:
        for line in record:
            count = 0
            if 'form' in "".join(line.lower().split(' ')):
                pass
            else:
                for pattern in patterns:
                    if "".join(pattern.lower().split(' ')) in "".join(line.lower().split(' ')):
                        count += 1
                if count == 1:
                    for pattern in patterns:
                        if "".join(pattern.lower().split(' ')) in "".join(line.lower().split(' ')):
                            single_pattern(record, line, pattern)
                if count > 1:
                    symbol = [':', '-', '=', '~']
                    keys = []
                    for pattern in patterns:
                        if "".join(pattern.lower().split(' ')) in "".join(line.lower().split(' ')):
                            keys.append(pattern)
                    values = ("".join(line.lower().split(keys[0].lower()))).split(keys[1].lower())[-2:]
                    for ind in range(len(values)):
                        for j in symbol:
                            values[ind] = values[ind].replace(j, '').strip()
                    if "".join(keys[0].lower().split(' ')) in ["mobilenumber", "email"]:
                        d1.setdefault(keys[0].title(), []).append(("".join(values[0].split(" "))).lower())
                    else:
                        d1.setdefault(keys[0].title(), []).append(values[0].strip())

                    if "".join(keys[1].lower().split(' ')) in ["mobilenumber", "email"]:
                        d1.setdefault(keys[1].title(), []).append(("".join(values[1].split(" "))).lower())
                    else:
                        d1.setdefault(keys[1].title(), []).append(values[1].strip())


def data_store():
    df = pd.DataFrame(d1)
    df.to_excel(r'records.xlsx', index=None, header=True)
    print(df)


def analysis():
    lables = ['Age', 'Rating', 'Mark', 'Percentage', 'Score']
    df = pd.read_excel('records.xlsx')
    for column in df.columns:
        for lable in lables:
            if column == lable:
                df[column].plot(kind="hist")
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.savefig(r'%s.png' % column)


@app.route('/', methods=['GET', 'POST'])
def homeegh():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    mssg = ""
    if request.method == 'POST' and 'uname' in request.form and 'pwd' in request.form:
        # Create variables for easy access
        username = request.form['uname']
        password = request.form['pwd']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM reg1 WHERE username = %s AND password = %s', [username, password])
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['uname'] = account['username']
            global sessionname
            sessionname = session['uname']

            # Redirect to home page
            mssg = ""
            return redirect('/start')

        else:
            # Account doesnt exist or username/password incorrect
            mssg = "*Incorrect username/password"
            return render_template('login.html', msg=mssg)

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'uname' in request.form and 'pwd' in request.form and 'emailid' in request.form and 'pwd' in request.form and 'mno' in request.form:
        # Create variables for easy access
        username = request.form['uname']
        password = request.form['pwd']
        email = request.form['emailid']
        pwd = request.form['pwd']
        cpwd = request.form['cpwd']
        mno = request.form['mno']
        print(username)
        print(email)
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM reg1 WHERE username = %s', [username])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        elif cpwd != pwd:
            msg = 'please enter same password'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO reg1 VALUES (NULL, %s, %s, %s,%s)', (username, email, cpwd, mno))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            return render_template('register.html', msg=msg)
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        # Show registration form with message (if any)
        return render_template('register.html', msg=msg)
    return render_template('register.html', msg=msg)


@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template('second.html')


@app.route('/home', methods=['GET', 'POST'])
def home1():
    if session['loggedin'] == True:
        if request.method == 'POST':
            a1 = (request.form['abc'])
            a1 = a1.replace('[', '').replace(']', '').replace('"', '')
            patterns.extend(a1.split(','))
            print(patterns)
            return render_template('home.html')

        return render_template('home.html')


@app.route('/homee', methods=['GET', 'POST'])
def index():
    if "file_urls" not in session:
        session['file_urls'] = []

    # list to hold our uploaded image urls

    file_urls = session['file_urls']
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # save the file with to our photos folder
            photos.save(file, name=file.filename)
            filenamej = file.filename
            file_urls.append(photos.url(filenamej))
            # count = count + 1
            session['file_urls'] = file_urls

        return "uploading..."
    return render_template('home.html')


# Processing and Extracting Multiple Images For generating Text

@app.route('/showdata', methods=['GET', 'POST'])
def ProcessImage():
    if session['loggedin'] == True:
        maintext = ""
        imgnames = sorted(glob("\Images\*.*"))
        # Read the image into a byte array
        for images in imgnames:
            image_data = open(images, "rb").read()

            headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
            # put the byte array into your post request
            response = requests.post(text_recognition_url, headers=headers, data=image_data)


            # The recognized text isn't immediately available, so poll to wait for completion.
            analysis = {}

            poll = True
            while (poll):
                response_final = requests.get(
                response.headers["Operation-Location"], headers=headers)
                analysis = response_final.json()
                # print(analysis)
                time.sleep(1)
                if "recognitionResults" in analysis:
                    poll = False
                if "status" in analysis and analysis['status'] == 'Failed':
                    poll = False

            polygons = []
            if "recognitionResults" in analysis:
                # count += 1;
                # Extract the recognized text, with bounding boxes.
                polygons = [(line["boundingBox"], line["text"])
                            for line in analysis["recognitionResults"][0]["lines"]]
                # print(polygons)
                text = []
                for i in polygons:
                    text.append(i[1])

                records.append(text)

                f = open("ImageExtractionn.txt", "a")
                for line in text:
                    f.write(line + '\n')
                f.close()
            global sessionname, cursor
            print(patterns)
            print(records)
            # analysis()
            file = open('ImageExtractionn.txt', 'r')
            file_content = file.read()
            file.close()

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('delete from download where username=%s', [sessionname])
            mysql.connection.commit()
            cursor.execute('INSERT INTO download VALUES (NULL, %s, %s)', [sessionname, file_content])
            mysql.connection.commit()
        os.remove(r'\ImageExtractionn.txt')
        data_cleaning(records)
        data_store()
        filelist = glob(os.path.join(r'\Images\*.*'))
        for f in filelist:
            os.remove(f)
        cursor.execute('select txt_file from download where username=%s', [sessionname])
        sql = "SELECT txt_file from download where username='" + sessionname + "';"
        cursor.execute(sql)
        d = cursor.fetchall()
        f1 = open("\download.txt", "w")
        f1.write(str(d))
        f1.close()
        print(d)
        b2 = os.path.getsize(r'\download.txt')
        b1 = os.path.getsize(r'\records.xlsx')

        def sizecount(b):
            global a
            co = 0
            while (b > 1024):
                b = b / 1024
                co = co + 1
            if (co == 0):
                a = str(b) + " Bytes"
            if (co == 1):
                a = str(b) + " KB"
            if (co == 2):
                a = str(b) + " MB"
            if (co == 3):
                a = str(b) + " GB"
            if (co == 4):
                a = str(b) + " TB"
            return a

        a1 = sizecount(b1)
        a2 = sizecount(b2)
        return render_template('showdata.html', name=a2, name2=a1)

    flag = False


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if session['loggedin'] == True:
        df0 = pd.read_excel(r'\records.xlsx')
        Mytest(df0)
        global plot_url
#        return render_template('prediction.html', url=plot_url)
        return render_template('prediction.html',url=plot_url)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if session['loggedin'] == True:
        if request.method == 'POST':
            f = request.files['file']
            fname = "uploadFile.xlsx"
            f.save(os.path.join(UPLOAD_FOLDER, fname))
            df1 = pd.read_excel(r'\uploadFile.xlsx')
            Mytest(df1)
            return render_template('prediction.html',url=plot_url)
    return render_template('prediction1.html',url=plot_url)


@app.route('/download', methods=['GET', 'POST'])
def download():
    path = "\download.txt"
    return send_file(path, as_attachment=True)


@app.route('/download1', methods=['GET', 'POST'])
def download1():
    path = r'\records.xlsx'
    return send_file(path, as_attachment=True)


@app.route("/download4", methods=['GET', 'POST'])
def download4():
    path = r'\positive_prediction.xlsx'
    return send_file(path, as_attachment=True)

@app.route("/download3", methods=['GET', 'POST'])
def download3():
    path = r'\negative_prediction.xlsx'
    return send_file(path, as_attachment=True)

@app.route("/send1Email",methods=['GET','POST'])
def send1Email1():
    return render_template('sendEmail.html')


@app.route("/send2Email",methods=['GET','POST'])
def send2Email():
    if request.method=='POST':
        import pandas as pd
        import csv, smtplib, ssl
        dic2 = {}
        df = pd.read_excel(r'\positive_prediction.xlsx')
        dic2['Name'] = df['Name']
        dic2['Email']=df['Email']
        df2 = pd.DataFrame(dic2)
        df2.to_csv(r'\positivesend.csv')

        with open(r'\positivesend.csv') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for A, Name, Email in reader:
                print(f"Sending email to {Name}")
                # Send email here
        message = request.form['emailMsg'];
        from_address = ""
        password = ""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(from_address, password)
            with open(r'\positivesend.csv') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for A, Name, Email in reader:
                    server.sendmail(
                        from_address,
                        Email,
                        message.format(Name=Name),
                    )
                    msg="successfully done"
                    return render_template('sendEmail.html',msg=msg)
    msg="You haven't done the proccess"
    return render_template('sendEmail.html',msg=msg)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    d1.clear()
    records.clear()
    patterns.clear()
    session['loggedin'] = False
    session.pop('loggedin', None)
    return redirect('/login')


if __name__ == "__main__":
    app.run(debug=True)
