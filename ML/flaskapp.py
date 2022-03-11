from flask import Flask, render_template, Response
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    model = tf.keras.models.load_model('newds_model.h5')
    pred_list = ['Backward', 'Forward', 'Left', 'Right', 'Stop']
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            # frame.shape = (480, 640, 3) for camera used. Other cams can be used as well
            
            #Copying unmodified frame to return as response to video stream
            iframe = frame.copy()
            
            
            # Processing the images/frames
            # frame is of type numpy.ndarray
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (150, 150))
            blur = cv2.GaussianBlur(frame, (31, 31), 0)
            ret, frame = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            frame = frame.astype(float)
            frame = frame / 255.
            frame = frame.reshape(150, 150, 1)
            
            
            # Make prediction using loaded model
            
            prediction = model.predict(np.array([frame]))
            pred_index = np.argmax(prediction)
            pred_char = pred_list[pred_index]
            cv2.putText(iframe, (pred_char + ' ' + str(np.max(prediction))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            # Sending back response and prediction to video feed
            
            ret, buffer = cv2.imencode('.jpg', iframe)
            iframe = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + iframe + b'\r\n')

            
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
