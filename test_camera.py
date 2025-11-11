import cv2
print("OpenCV version:", cv2.__version__)
print("Attempting to open camera...")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera {i} works!")
        ret, frame = cap.read()
        print(f"  Frame captured: {ret}")
        if ret:
            print(f"  Frame shape: {frame.shape}")
        cap.release()
    else:
        print(f"✗ Camera {i} failed")