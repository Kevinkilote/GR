from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/detect/bdd100k_yolov83/weights/best.pt')

# Load video
cap = cv2.VideoCapture("carla_videos/carla_hard_rainy_night.mp4")

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 1280, 720)

paused = False
frame_pos = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        results = model.predict(source=frame, conf=0.35, save=False, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Detections", annotated)

    key = cv2.waitKey(0 if paused else 1) & 0xFF

    if key == 27 or key == ord('x'):  # ESC or 'x' to quit
        break
    elif key == 32:  # Spacebar to pause/resume
        paused = not paused
    elif key == ord('e'):  # 'E' to go forward 1 frame
        paused = True
        frame_pos += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            results = model.predict(source=frame, conf=0.3, save=False, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("Detections", annotated)
    elif key == ord('q'):  # 'Q' to go backward 1 frame
        paused = True
        frame_pos = max(0, frame_pos - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            results = model.predict(source=frame, conf=0.3, save=False, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("Detections", annotated)

cap.release()
cv2.destroyAllWindows()
