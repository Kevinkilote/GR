import argparse
import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections on a video file.")
    parser.add_argument("video", type=str, help="Path to the input video.")
    parser.add_argument("--model", type=str, default="runs/detect/train3/weights/best.pt", help="Path to the YOLO model weights.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detections.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--display-size", type=str, default=None, help="Optional display size WIDTHxHEIGHT.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device {device}")

    print(f"INFO: Loading YOLO model from '{args.model}'")
    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Unable to open video '{args.video}'")
        return

    display_dims = None
    if args.display_size:
        try:
            width, height = map(int, args.display_size.lower().split("x"))
            display_dims = (width, height)
        except ValueError:
            print("ERROR: --display-size must be WIDTHxHEIGHT, e.g., 1280x720")
            return

    print("INFO: Press 'q' to quit, spacebar to pause/resume.")
    paused = False
    annotated_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("INFO: End of video.")
                break

            results = model(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            annotated_frame = results[0].plot()
        elif annotated_frame is None:
            continue

        frame_to_show = annotated_frame
        if display_dims:
            frame_to_show = cv2.resize(frame_to_show, display_dims)

        cv2.imshow("YOLOv11 Video Demo", frame_to_show)
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
