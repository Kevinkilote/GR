import argparse
from typing import Dict, Optional, Set, Tuple

import cv2
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms

# --- SCRIPT CONTROLS ---
#  - Press 'spacebar' to pause/resume the video.
#  - Press 'q' to quit the video player.

# --- CONFIGURATION ---
# These class names must match the ResNet training order exactly.
RESNET_CLASS_NAMES = [
    'information--parking--g1',
    'information--pedestrians-crossing--g1',
    'information--tram-bus-stop--g2',
    'regulatory--go-straight--g1',
    'regulatory--keep-right--g1',
    'regulatory--maximum-speed-limit-40--g1',
    'regulatory--no-entry--g1',
    'regulatory--no-left-turn--g1',
    'regulatory--no-parking--g1',
    'regulatory--no-stopping--g15',
    'regulatory--no-u-turn--g1',
    'regulatory--priority-road--g4',
    'regulatory--stop--g1',
    'regulatory--yield--g1',
    'warning--children--g2',
    'warning--curve-left--g2',
    'warning--pedestrians-crossing--g4',
    'warning--road-bump--g2',
    'warning--slippery-road-surface--g1'
]

# Lower-case names of YOLO classes that should be refined by the ResNet recognizer.
# Update this list if your YOLO checkpoint exposes additional traffic-sign labels.
DEFAULT_SIGN_LABELS = {
    'traffic sign',
}

# Colors for bounding boxes (BGR).
BOX_COLORS = {
    'traffic sign': (0, 255, 0),     # green
    'traffic light': (0, 255, 255),  # yellow
    'car': (255, 0, 0),              # blue
    'default': (255, 255, 255),      # white
}


def parse_display_size(arg: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse WIDTHxHEIGHT strings provided on the CLI."""
    if not arg:
        return None
    try:
        width, height = map(int, arg.lower().split('x'))
        return width, height
    except ValueError:
        raise ValueError("--display-size must use WIDTHxHEIGHT format, e.g., 1280x720")


def parse_sign_labels(raw_labels: Optional[str]) -> Optional[Set[str]]:
    """Convert a comma-separated label string into a normalized set."""
    if not raw_labels:
        return None
    labels = {label.strip().lower() for label in raw_labels.split(',') if label.strip()}
    return labels or None


def load_models(yolo_path: str, resnet_path: str, device: torch.device):
    """Load YOLO and ResNet models and return them with YOLO label mapping."""
    print(f"INFO: Loading YOLO detector from '{yolo_path}'...")
    yolo_model = YOLO(yolo_path).to(device)

    names_attr = yolo_model.names
    if isinstance(names_attr, dict):
        yolo_class_names: Dict[int, str] = names_attr
    else:
        yolo_class_names = {idx: name for idx, name in enumerate(names_attr)}
    print("INFO: YOLO model loaded successfully.")

    print(f"INFO: Loading ResNet recognizer from '{resnet_path}'...")
    resnet_model = torchvision.models.resnet18(weights=None)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_ftrs, len(RESNET_CLASS_NAMES))
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    print("INFO: ResNet model loaded successfully.")

    return yolo_model, yolo_class_names, resnet_model


def classify_sign(
    frame,
    bbox,
    resnet_model,
    resnet_transform,
    device,
) -> Optional[Tuple[str, float]]:
    """Crop a detection, run it through ResNet, and return name/confidence."""
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    input_tensor = resnet_transform(cropped_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        predicted_idx = torch.argmax(probabilities).item()
        sign_name = RESNET_CLASS_NAMES[predicted_idx]

    return sign_name, confidence


def main(
    video_path: str,
    yolo_model_path: str,
    resnet_model_path: str,
    display_size: Optional[Tuple[int, int]],
    sign_label_overrides: Optional[Set[str]],
    conf_threshold: float,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")

    try:
        yolo_model, yolo_class_names, resnet_model = load_models(
            yolo_model_path, resnet_model_path, device
        )
    except Exception as exc:
        print(f"ERROR: Could not load models. {exc}")
        return

    sign_labels = sign_label_overrides or DEFAULT_SIGN_LABELS
    print("INFO: YOLO classes that will be refined by ResNet:")
    for name in sorted(sign_labels):
        print(f"      - {name}")

    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at '{video_path}'")
        return

    paused = False
    frame = None

    print("\nINFO: Starting video processing...")
    print("      Press 'spacebar' to pause/play.")
    print("      Press 'q' to quit.")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("INFO: End of video reached.")
                break

            results = yolo_model(frame, conf=conf_threshold, verbose=False)[0]

            for box in results.boxes:
                score = float(box.conf.item())
                class_id = int(box.cls.item())
                class_name = yolo_class_names.get(class_id, f"class_{class_id}")
                class_name_lower = class_name.lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                label = f"{class_name}: {score:.2f}"
                color = BOX_COLORS.get(class_name_lower, BOX_COLORS['default'])

                if class_name_lower in sign_labels and score >= conf_threshold:
                    classified = classify_sign(
                        frame, (x1, y1, x2, y2), resnet_model, resnet_transform, device
                    )
                    if classified is not None:
                        sign_name, confidence = classified
                        label = f"{sign_name}: {confidence:.2f}"
                        color = BOX_COLORS['traffic sign']

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        if frame is None:
            break

        frame_to_display = frame
        if display_size:
            frame_to_display = cv2.resize(frame_to_display, display_size)

        cv2.imshow('Traffic Sign Recognition Demo', frame_to_display)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
            print("INFO: Paused. Press spacebar to resume." if paused else "INFO: Resumed.")

    cap.release()
    cv2.destroyAllWindows()
    print("INFO: Demo finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect cars, traffic lights, and traffic signs with YOLO, refine signs with ResNet."
    )
    parser.add_argument('--video', type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument('--yolo', type=str, default='runs/detect/train3/weights/best.pt', help="Path to the YOLO .pt model file.")
    parser.add_argument('--resnet', type=str, default='best_traffic_sign_classifier_advanced.pth', help="Path to the ResNet .pth model file.")
    parser.add_argument('--display-size', type=str, default=None, help="Optional display window size, e.g., '1280x720'.")
    parser.add_argument(
        '--sign-classes',
        type=str,
        default=None,
        help="Comma-separated YOLO class names that should be refined with the traffic-sign classifier.",
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.35,
        help="Confidence threshold to filter YOLO detections before visualizing.",
    )

    args = parser.parse_args()

    try:
        display_dimensions = parse_display_size(args.display_size)
    except ValueError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)

    override_sign_labels = parse_sign_labels(args.sign_classes)

    main(
        args.video,
        args.yolo,
        args.resnet,
        display_dimensions,
        override_sign_labels,
        args.conf,
    )
