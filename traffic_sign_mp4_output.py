import cv2
import torch
import torchvision
import numpy as np
import argparse
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import os

# --- SCRIPT CONTROLS ---
#  - Press 'spacebar' to pause/resume the video.
#  - Press 'q' to quit the video player.

# --- CONFIGURATION ---
# This list MUST match the classes your ResNet model was trained on.
# Note: 'warniing--curve-left--g2' is kept as provided, ensure it matches your folder name.
CLASS_NAMES = [
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

def main(video_path, yolo_model_path, resnet_model_path, display_size):
    """
    Main function to process a video file and save the output with detections.
    """
    # --- MODEL AND DEVICE SETUP ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"INFO: Using device: {device}")

        print(f"INFO: Loading YOLO detector from '{yolo_model_path}'...")
        yolo_model = YOLO(yolo_model_path).to(device)
        print("INFO: YOLO model loaded successfully.")

        print(f"INFO: Loading ResNet recognizer from '{resnet_model_path}'...")
        resnet_model = torchvision.models.resnet18(weights=None)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
        print("INFO: ResNet model loaded successfully.")

    except Exception as e:
        print(f"ERROR: Could not load models. {e}")
        return

    # --- IMAGE TRANSFORMATION SETUP ---
    resnet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- VIDEO INPUT SETUP ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at '{video_path}'")
        return

    # --- MODIFICATION: VIDEO OUTPUT SETUP ---
    # Get video properties for the output file.
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the output filename.
    file_name, file_ext = os.path.splitext(video_path)
    output_path = f"{file_name}_processed.mp4"
    
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
    print(f"INFO: Saving processed video to '{output_path}'")
    # --- END MODIFICATION ---

    paused = False
    print("\nINFO: Starting video processing...")
    print("      Press 'spacebar' to pause/play.")
    print("      Press 'q' to quit.")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("INFO: End of video reached.")
                break

            # --- AI INFERENCE PIPELINE ---
            detections = yolo_model(frame, verbose=False)[0]

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                
                if score > 0.65:
                    cropped_sign_np = frame[int(y1):int(y2), int(x1):int(x2)]
                    cropped_sign_pil = Image.fromarray(cv2.cvtColor(cropped_sign_np, cv2.COLOR_BGR2RGB))
                    input_tensor = resnet_transform(cropped_sign_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence = torch.max(probabilities).item()
                        predicted_idx = torch.argmax(probabilities).item()
                        sign_name = CLASS_NAMES[predicted_idx]

                    # --- VISUALIZATION ---
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{sign_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- DISPLAY FRAME AND HANDLE USER INPUT ---
        display_frame = cv2.resize(frame, display_size) if display_size else frame
        cv2.imshow('Traffic Sign Recognition Demo', display_frame)

        # --- MODIFICATION: WRITE FRAME TO OUTPUT VIDEO ---
        # Write the original, un-resized frame with annotations to the file.
        video_writer.write(frame)
        # --- END MODIFICATION ---

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
            print("INFO: Paused." if paused else "INFO: Resumed.")

    # --- CLEANUP ---
    cap.release()
    video_writer.release() # Release the video writer object.
    cv2.destroyAllWindows()
    print(f"INFO: Demo finished. Processed video saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video to detect and recognize traffic signs for a demo.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input MP4 video file.")
    parser.add_argument('--yolo', type=str, default='runs/detect/train3/weights/best.pt', help="Path to the YOLO .pt model file.")
    parser.add_argument('--resnet', type=str, default='best_traffic_sign_classifier_advanced.pth', help="Path to the ResNet .pth model file.")
    parser.add_argument('--display-size', type=str, default=None, help="Optional display window size, e.g., '1280x720'.")
    
    args = parser.parse_args()

    display_dimensions = None
    if args.display_size:
        try:
            width, height = map(int, args.display_size.split('x'))
            display_dimensions = (width, height)
        except ValueError:
            print("ERROR: Invalid format for --display-size. Please use WIDTHxHEIGHT, e.g., '1280x720'.")
            exit()
    
    main(args.video, args.yolo, args.resnet, display_dimensions)
