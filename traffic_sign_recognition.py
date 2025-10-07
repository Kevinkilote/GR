"""Shared configuration and helpers for traffic sign recognition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torchvision
from torchvision import transforms

# These class names must match the ResNet training order exactly.
RESNET_CLASS_NAMES: Sequence[str] = (
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
    'warning--slippery-road-surface--g1',
)

# Lower-case YOLO class names that should be refined by the ResNet recogniser by default.
DEFAULT_SIGN_LABELS = frozenset({'traffic sign'})
OTHER_SIGN_LABEL = 'other-sign'

# Normalisation applied during training.
RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass(frozen=True)
class ResNetBundle:
    """Container for a loaded ResNet model and its device/transform."""

    model: torch.nn.Module
    device: torch.device
    transform: transforms.Compose


def load_resnet(weights_path: str, *, device_hint: Optional[str] = None) -> ResNetBundle:
    """Load the fine-tuned ResNet18 checkpoint used for sign recognition."""
    if device_hint:
        device = torch.device(device_hint)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(RESNET_CLASS_NAMES))

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return ResNetBundle(model=model, device=device, transform=RESNET_TRANSFORM)


def normalise_sign_labels(labels: Optional[Iterable[str]]) -> Iterable[str]:
    """Normalise user-provided sign labels to lowercase values."""
    if labels is None:
        return DEFAULT_SIGN_LABELS
    return {label.strip().lower() for label in labels if label}
