from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import json
import sys

app = Flask(__name__)

# Known constants for distance calculation
KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 14.0
FOCAL_LENGTH = None

# Distance thresholds
OPTIMAL_DISTANCE_MIN = 50  # cm
OPTIMAL_DISTANCE_MAX = 70  # cm

# Initialize face and eye cascade classifiers
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("✓ Cascade classifiers loaded successfully")
except Exception as e:
    print(f"✗ Error loading cascade classifiers: {e}")
    sys.exit(1)

# Database initialization
def init_db():
    conn = sqlite3.connect('eyesight_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  distance REAL,
                  eye_distance REAL,
                  posture_status TEXT)''')
    conn.commit()
    conn.close()
    print("✓ Database initialized")

init_db()

class VideoCamera:
    def __init__(self):
        print("Attempting to open camera...")
        # Release any existing camera first
        self.video = None
        
        # Try to open camera with retries
        for attempt in range(3):
            self.video = cv2.VideoCapture(0)
            if self.video.isOpened():
                break
            print(f"Attempt {attempt + 1} failed, retrying...")
            if self.video:
                self.video.release()
            import time
            time.sleep(0.5)
        
        if not self.video.isOpened():
            print("✗ ERROR: Could not open camera!")
            print("Please check:")
            print("  1. Camera is connected")
            print("  2. No other app is using the camera")
            print("  3. Camera permissions are granted")
        else:
            print("✓ Camera opened successfully")
            # Set camera properties for better performance
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.current_distance = 0
        self.eye_distance = 0
        self.posture_status = "Unknown"
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
            print("✓ Camera released")
    
    def is_opened(self):
        return self.video.isOpened()
    
    def calculate_distance(self, face_width_pixels):
        """Calculate distance using focal length"""
        global FOCAL_LENGTH
        if FOCAL_LENGTH is None:
            # Initial calibration - assume user is at optimal distance
            FOCAL_LENGTH = (face_width_pixels * KNOWN_DISTANCE) / KNOWN_WIDTH
        
        if face_width_pixels > 0:
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / face_width_pixels
            return distance
        return 0
    
    def get_posture_status(self, distance):
        """Determine posture status based on distance"""
        if distance == 0:
            return "No face detected"
        elif distance < OPTIMAL_DISTANCE_MIN:
            return "Too close"
        elif distance > OPTIMAL_DISTANCE_MAX:
            return "Too far"
        else:
            return "Optimal"
    
    def get_frame(self):
        if not self.video.isOpened():
            # Return a black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Not Available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        
        success, frame = self.video.read()
        if not success:
            print("✗ Failed to read frame from camera")
            # Return a black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Failed to Read Camera", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.current_distance = 0
            self.posture_status = "No face detected"
            # Add instructions on frame
            cv2.putText(frame, "Position your face in front of camera", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Calculate distance
            self.current_distance = self.calculate_distance(w)
            
            # Get posture status
            self.posture_status = self.get_posture_status(self.current_distance)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            # Draw rectangles around eyes and calculate distance between them
            if len(eyes) >= 2:
                # Sort eyes by x coordinate
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                eye1 = eyes_sorted[0]
                eye2 = eyes_sorted[1]
                
                # Draw eye rectangles
                for (ex, ey, ew, eh) in [eye1, eye2]:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Calculate distance between eye centers
                eye1_center = eye1[0] + eye1[2]//2
                eye2_center = eye2[0] + eye2[2]//2
                self.eye_distance = abs(eye2_center - eye1_center)
                
                # Draw line between eyes
                cv2.line(roi_color, 
                        (eye1_center, eye1[1] + eye1[3]//2),
                        (eye2_center, eye2[1] + eye2[3]//2),
                        (0, 255, 255), 2)
            
            # Display distance info on frame
            color = (0, 255, 0) if self.posture_status == "Optimal" else (0, 0, 255)
            cv2.putText(frame, f"Distance: {self.current_distance:.1f} cm", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, self.posture_status, 
                       (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Encode frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

# Initialize camera
print("\n" + "="*50)
print("Starting Eyesight Monitor Application")
print("="*50 + "\n")

# Initialize camera (but not in debug reload)
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

def gen_frames():
    cam = get_camera()
    while True:
        frame = cam.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    cam = get_camera()
    return jsonify({
        'is_opened': cam.is_opened(),
        'message': 'Camera is working' if cam.is_opened() else 'Camera not available'
    })

@app.route('/get_measurements')
def get_measurements():
    cam = get_camera()
    return jsonify({
        'distance': round(cam.current_distance, 2),
        'eye_distance': cam.eye_distance,
        'posture_status': cam.posture_status
    })

@app.route('/save_measurement', methods=['POST'])
def save_measurement():
    data = request.json
    conn = sqlite3.connect('eyesight_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO measurements (timestamp, distance, eye_distance, posture_status)
                 VALUES (?, ?, ?, ?)''',
              (datetime.now().isoformat(),
               data['distance'],
               data['eye_distance'],
               data['posture_status']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/get_history')
def get_history():
    conn = sqlite3.connect('eyesight_data.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, distance, eye_distance, posture_status 
                 FROM measurements ORDER BY timestamp DESC LIMIT 50''')
    rows = c.fetchall()
    conn.close()
    
    history = [{
        'timestamp': row[0],
        'distance': row[1],
        'eye_distance': row[2],
        'posture_status': row[3]
    } for row in rows]
    
    return jsonify(history)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """Recalibrate the distance measurement"""
    global FOCAL_LENGTH
    data = request.json
    actual_distance = data.get('distance', KNOWN_DISTANCE)
    
    cam = get_camera()
    # Get current face width in pixels
    success, frame = cam.video.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            FOCAL_LENGTH = (w * actual_distance) / KNOWN_WIDTH
            print(f"✓ Calibrated: Focal length = {FOCAL_LENGTH:.2f}")
            return jsonify({'status': 'success', 'focal_length': FOCAL_LENGTH})
    
    return jsonify({'status': 'error', 'message': 'No face detected'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)