from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
from scipy.spatial import distance as dist
import os
import logging

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

logging.basicConfig(level=logging.INFO)

def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

def analyze_video(video_path):
    magnification_map = {400: 0.234, 800: 0.117, 1600: 0.0586}
    magnification = 400
    micrometers_per_pixel = magnification_map.get(magnification, None)
    if not micrometers_per_pixel:
        return {"error": "Invalid magnification value."}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    speed_threshold, sperm_count_threshold = 10, 20

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    frame_count, total_sperm_count, speeds = 0, 0, []
    previous_positions = []
    kalman = initialize_kalman()
    previous_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_positions = []

        for contour in contours:
            if cv2.contourArea(contour) > 10:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    current_positions.append((cx, cy))

        if previous_frame is not None and previous_positions:
            optical_flow = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY),
                gray,
                np.float32(previous_positions),
                None
            )[0]
            for curr_pos, flow in zip(previous_positions, optical_flow):
                speed = dist.euclidean(curr_pos, flow) * micrometers_per_pixel * fps
                speeds.append(speed)

        total_sperm_count += len(current_positions)
        frame_count += 1
        previous_positions = current_positions
        previous_frame = frame.copy()

    cap.release()

    average_speed = np.mean(speeds) if speeds else 0
    average_sperm_count = total_sperm_count / frame_count if frame_count else 0
    is_normal = average_speed >= speed_threshold and average_sperm_count >= sperm_count_threshold

    return {
        "average_speed": round(average_speed, 2),
        "average_count": round(average_sperm_count, 2),
        "is_normal": is_normal
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index.html', error="No video file selected.")

        video_file = request.files['video']
        if video_file.filename == '':
            return render_template('index.html', error="No video file selected.")

        # Save the uploaded file to a temporary directory
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        video_path = os.path.join(upload_folder, video_file.filename)
        video_file.save(video_path)

        # Analyze the uploaded video
        results = analyze_video(video_path)
        os.remove(video_path)  # Remove the file after analysis

        if "error" in results:
            return render_template('index.html', error=results["error"])

        return render_template('index.html', results=results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
