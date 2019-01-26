from pydarknet import Detector, Image
import cv2
import os
import time

from flask import Flask, jsonify, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

# Load Detectors
net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))


ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'mp4']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            prefix = os.urandom(32).hex() + '-'
            filename = prefix + secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = 'out-' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            if file.filename.endswith('mp4'):
                # VIDEO
                res = scan_vid(input_path, output_path)
                return jsonify({
                    'results': res['results'],
                    # 'time': res['time'],
                    'input': url_for('uploaded_file', filename=filename),
                    'output': url_for('uploaded_file', filename=output_filename)
                })
            else:
                # PHOTOS
                res = scan(input_path, output_path)
                return jsonify({
                    'results': res['results'],
                    'time': res['time'],
                    'input': url_for('uploaded_file', filename=filename),
                    'output': url_for('uploaded_file', filename=output_filename)
                })
    return '''
    <!doctype html>
    <title>Analyze new File</title>
    <h1>Issou X YOLO: Analyze new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

from multiprocessing import Pool, cpu_count, current_process


def do_img(im):
    print('do_img', current_process())
    im2 = Image(im)
    results = net.detect(im2)
    results_output = []

    for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
            cv2.putText(im,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX,4,(0,0,255), thickness=2)
            results_output.append({
                'cat': cat.decode(),
                'score': score,
                'bounds': {
                    'x': bounds[0],
                    'y': bounds[1],
                    'w': bounds[2],
                    'h': bounds[3]
                }
            })
    print('do_img end', current_process())
    return (im, results_output)

def scan_vid(input_path, output_path):
    vid = cv2.VideoCapture(input_path)
    outvid = None
    h, w, l = (None, None, None)

    results_vid = []

    imgs = []
    

    while vid.isOpened():
        success, im = vid.read()
        if success:
            imgs.append(im)
        else:
            break
    if not outvid:
        h, w, l = imgs[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outvid = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        print('videowriter', w, h)
        print(len(imgs), 'frames')
    with Pool(processes=1) as p:
        imgs_res = p.map(do_img, imgs)
    
        for ires in imgs_res:
            outvid.write(ires[0])
            results_vid.append(ires[1])

        vid.release()
        outvid.release()

        return {
            'results': results_vid
        }

def scan(input_path, output_path):
    img = cv2.imread(input_path)
    img2 = Image(img)

    start_time = time.time()
    results = net.detect(img2)
    end_time = time.time()

    results_output = []

    for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
            cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255), thickness=2)
            results_output.append({
                'cat': cat.decode(),
                'score': score,
                'bounds': {
                    'x': bounds[0],
                    'y': bounds[1],
                    'w': bounds[2],
                    'h': bounds[3]
                }
            })
    cv2.imwrite(output_path, img)

    return {
        'time': end_time-start_time,
        'results': results_output
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=13131)
