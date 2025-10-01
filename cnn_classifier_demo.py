import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

# --- CONFIGURATION ---
# This list MUST match the 19 class names your ResNet model was trained on,
# in the exact same order.
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
    'warniing--curve-left--g2',
    'warning--pedestrians-crossing--g4',
    'warning--road-bump--g2',
    'warning--slippery-road-surface--g1'
]

# --- PATH CONFIGURATION ---
# Manually edit the paths below to point to your files.
MODEL_PATH = 'best_traffic_sign_classifier_advanced.pth'
# IMPORTANT: Change this path to the image you want to test.
IMAGE_TO_TEST = 'cnn_sign_dataset/test/regulatory--no-parking--g1/1D4_jlrlqZ9BT3bcq4VjCg_hr2gmfjltgns3cia8v84l4rtr4.jpg' 

def test_classifier(model_path, image_path):
    """
    Loads a ResNet model and classifies a single image.
    """
    # --- MODEL AND DEVICE SETUP ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"INFO: Using device: {device}")

        print(f"INFO: Loading ResNet recognizer from '{model_path}'...")
        model = torchvision.models.resnet18(weights=None)
        
        # IMPORTANT: The number of output features must match the number of classes the model was trained on.
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("INFO: ResNet model loaded successfully.")

    except Exception as e:
        print(f"ERROR: Could not load the model. {e}")
        return

    # --- IMAGE PREPROCESSING (THE FIX) ---
    # This transformation pipeline MUST MATCH the 'val' pipeline from your training script.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        print(f"INFO: Loading and preprocessing image: '{image_path}'")
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        print("INFO: Image preprocessed successfully.")

    except FileNotFoundError:
        print(f"ERROR: Cannot find the image at '{image_path}'.")
        return
    except Exception as e:
        print(f"ERROR: Could not load or process the image. {e}")
        return

    # --- INFERENCE ---
    with torch.no_grad():
        print("\nINFO: Running inference...")
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

    # --- RESULTS ---
    print("\n--- CLASSIFICATION RESULT ---")
    print(f"Predicted Sign: {predicted_class}")
    print(f"Confidence:     {confidence_score:.4f}")
    print("---------------------------\n")


if __name__ == '__main__':
    test_classifier(MODEL_PATH, IMAGE_TO_TEST)
