
"""
Feature Extraction

1. Downloads ResNet-50 + RadImageNet weights into ./weights (only happens once)
2. Loads the model as a feature extractor (removes the classification head)
3. Runs every slice from every .npy file through it to get a 2048-dim feature vector
"""

import torch
import torch.nn.functional as F
import numpy as np
import os

from config import CLUSTERING_FEATURES_PATH



class FeatureExtractor:


    def __init__(self, data_dir):
        """data_dir: The path to the .npy files (e.g ./MRNet-dataset/Files/train/axial)"""
        self.data_dir = data_dir

    # ---------------------------------------------------------------------------
    # Download and load ResNet-50 + RadImageNet weights
    # ---------------------------------------------------------------------------
    def _load_feature_extractor(self):
        """
        The class uses a CNN with ResNet-50 architecture trained on RadImageNet dataset.
        Downloads RadImageNet ResNet-50 weights (first run only) and returns a ResNet-50 with the classification head removed (calling model(X) returns a 2048-dim feature vector instead of class scores).
        """

        # Point the torch hub's download folder into the project instead of system cache.
        torch.hub.set_dir("./weights")

        # Downloading is done only the first time. After that it uses the cached files in ./weights.
        model = torch.hub.load("Warvito/radimagenet-models", "radimagenet_resnet50")

        # Replace the final classification layer with a dummy layer that just passes the pooled features through (f(x) = x)
        model.fc = torch.nn.Identity()

        # Turn off training behaviour and switch to evaluation (inference) mode
        model.eval()

        return model


    # ---------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------
    def _preprocess_slice(self, slice_2d):
        """
        ResNet-50 expects a 4D float tensor with mini-batches of normalized 3-channel (RGB) images.

        Shape: (B, C, H, W)
                B - Batch size (The number of images processed at once)
                C - Channels (Must be exactly 3 representing Red, Green, Blue)
                H - Height (Default resolution is 224x224)
                W - Widht (Default resolution is 224x224)

        Pixels: Raw pixel values (0 to 255) must be scaled down to a float in the range of 0.0 - 1.0

        RadImageNet constants: Because the model was trained on the RadImageNet datases, it expects the input
                               featurs to match the dataset's distribution. Each channel must be normalized with:
                               Mean: [0.485, 0.456, 0.406]
                               Standard Deviation: [0.229, 0.224, 0.225]
        """

        # Create a float tensor copy from the NumPy slice
        x = torch.tensor(slice_2d, dtype=torch.float32)

        # Scale pixel values to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # Add Batch and Channel dimensions for interpolation -> Shape: (1, 1, H, W)
        x = x.unsqueeze(0).unsqueeze(0)

        # Resize from raw size to 224x224 -> Shape: (1, 1, 224, 224)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Remove batch dimension to work on channels -> Shape: (1, 224, 224)
        x = x.squeeze(0)

        # Expand to 3 channels (RGB grayscale replication) -> Shape: (3, 224, 224)
        x = x.expand(3, -1, -1)

        # Normalize using ImageNet/RadImageNet standard statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std

        # Add back the Batch dimension required by ResNet-50 -> Shape: (1, 3, 224, 224)
        return x.unsqueeze(0)


    # ---------------------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------------------
    def _extract_features(self,model, npy_paths, case_ids, device="cpu"):
        """
            npy_paths: list of file paths, e.g. ["data/axial/1.npy", "data/axial/2.npy", ...]
            case_ids:  list of matching case identifiers, same length as npy_paths,
                       e.g. [1, 2, ...] — used so you can trace a feature back to its case

            Returns:
                features: numpy array of shape (total_slices, 2048)
                slice_case_ids: which case each row of `features` came from
        """

        model = model.to(device)
        features = []
        slice_case_ids = []

        # No batching is used, each slice gets processed one by one.
        with torch.no_grad(): # Disable history tracking becouse we are not training with backpropagation
            for path, case_id in zip(npy_paths, case_ids):
                volume = np.load(path)  # shape (S, 256, 256)

                for s in range(volume.shape[0]):
                    x = self.preprocess_slice(volume[s]).to(device) # (1, 2048, 1, 1)
                    feat = model(x).view(-1).cpu().numpy()  # (2048,)
                    features.append(feat)
                    slice_case_ids.append(case_id)

        return np.stack(features), np.array(slice_case_ids)


    # ---------------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------------

    def run(self):


        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        npy_paths = [os.path.join(self.data_dir, f) for f in sorted(os.listdir(self.data_dir))]
        case_ids = [f.replace(".npy", "") for f in sorted(os.listdir(self.data_dir))]

        print("Loading model...")
        model = self._load_feature_extractor()

        print(f"Extracting features from {len(npy_paths)} cases...")
        features, slice_case_ids = self._extract_features(model, npy_paths, case_ids, device=device)
        print(f"Extracted {features.shape[0]} slice features, each {features.shape[1]}-dim")

        # Save features so future runs can skip straight to loading them
        np.savez(CLUSTERING_FEATURES_PATH, features=features, case_ids=slice_case_ids)
        print(f"Saved features to {CLUSTERING_FEATURES_PATH}")


