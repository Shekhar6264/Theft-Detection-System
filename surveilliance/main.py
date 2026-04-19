from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import time

app = Flask(__name__)
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    return jsonify(camera.get_status())

def gen():
    while True:
        frame = camera.get_frame()

        if not frame:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
