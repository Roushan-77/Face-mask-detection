from flask import Flask, render_template, Response, redirect, url_for, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import os
import time

app = Flask(__name__)

model = load_model("model/mask_detector.h5")
labels = {0: "Mask", 1: "No Mask"}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists("violations"):
    os.makedirs("violations")

cap = cv2.VideoCapture(0)

camera_active = False

total_faces = 0
mask_count = 0
no_mask_count = 0
violation_count = 0

last_capture_time = 0
screenshot_cooldown = 3

last_detect_time = 0
detect_cooldown = 2

frame_count = 0
process_every = 3
last_results = []


def gen_frames():
    global total_faces, mask_count, no_mask_count, violation_count
    global last_capture_time, camera_active
    global last_detect_time
    global frame_count, last_results

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        if camera_active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            frame_count += 1

            if frame_count % process_every == 0:
                last_results = []

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (224, 224))
                    face = face / 255.0
                    face = np.reshape(face, (1, 224, 224, 3))

                    preds = model.predict(face, verbose=0)
                    label = np.argmax(preds)
                    confidence = np.max(preds)

                    current_time = time.time()

                    if current_time - last_detect_time > detect_cooldown:
                        total_faces += 1

                        if label == 1:
                            no_mask_count += 1
                        else:
                            mask_count += 1

                        last_detect_time = current_time

                    last_results.append((x, y, w, h, label, confidence))

                    if label == 1:
                        if current_time - last_capture_time > screenshot_cooldown:
                            filename = f"violations/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(filename, frame)
                            violation_count += 1
                            last_capture_time = current_time

            for (x, y, w, h, label, confidence) in last_results:
                text = f"{labels[label]} ({confidence*100:.1f}%)"

                if label == 1:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Camera Stopped", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    images = os.listdir("violations")
    return render_template("dashboard.html", images=images, total=total_faces, mask=mask_count, no_mask=no_mask_count)


@app.route('/violations/<filename>')
def violations_file(filename):
    return send_from_directory('violations', filename)


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/delete')
def delete():
    for file in os.listdir("violations"):
        os.remove(os.path.join("violations", file))
    return redirect(url_for('index'))


@app.route('/start')
def start():
    global camera_active
    camera_active = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global camera_active
    camera_active = False
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=False)   