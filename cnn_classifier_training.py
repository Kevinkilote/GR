import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = 'cnn_sign_dataset'
SAVE_MODEL_PATH = 'best_traffic_sign_classifier_advanced.pth'
NUM_EPOCHS = 30  # Increased epochs for better convergence
BATCH_SIZE = 32

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Main training loop with detailed logging and plotting.
    """
    since = time.time()
    
    # Store history for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                print(f'New best model saved to {SAVE_MODEL_PATH} with accuracy: {best_acc:.4f}')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history):
    """
    Generates and saves plots for training/validation loss and accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plotting training and validation accuracy
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plotting training and validation loss
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.savefig('training_plots.png')
    print("\nTraining history plots saved to 'training_plots.png'")
    plt.show()

if __name__ == '__main__':
    print("--- Advanced Traffic Sign Classifier Training ---")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Less aggressive crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if not os.path.isdir(DATASET_PATH):
        print(f"ERROR: Dataset directory not found at '{DATASET_PATH}'")
        exit()

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    train_classes = sorted(image_datasets['train'].classes)
    val_classes = sorted(image_datasets['val'].classes)
    if train_classes != val_classes:
        print("ERROR: Mismatch between training and validation classes.")
        exit()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Found {len(class_names)} matching classes.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- MODEL SETUP (FINE-TUNING) ---
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all the parameters in the model first
    for param in model_ft.parameters():
        param.requires_grad = False
        
    # Unfreeze the parameters of the final convolutional block (layer4)
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
        
    # Replace the final fully connected layer (it's unfrozen by default)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Create a list of parameters to optimize (only the unfrozen ones)
    params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
    print(f"INFO: Number of parameter sets to be trained: {len(params_to_update)}")

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # --- START TRAINING ---
    trained_model, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         num_epochs=NUM_EPOCHS)
                                         
    # --- PLOT RESULTS ---
    plot_training_history(history)
