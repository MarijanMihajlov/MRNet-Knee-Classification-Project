from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import gdown

from ImagePreProcess import PreProcess
from DataAugment import Augment
from DataSplit import DataSplit
from Metrics import Metrics


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_ROOT = Path("./Cache")
RADIMAGENET_WEIGHTS = "./weights/checkpoints/RadImageNet-ResNet50_notop.pth"


# ---------------------------------------------------------------------------
# Learning parameters
# ---------------------------------------------------------------------------
CNN_FEATURE_DIM = 2048
RADIOMICS_DIM = 45
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset — reads cached .npz, applies Augment (train only) + normalize_case (always)
# ---------------------------------------------------------------------------

class CacheDataset(Dataset):

    """
        Dataset class that reads, augments (only on train cases) and normalizes
        cleaned cases from the cache (already formed).

        Normalization is performed as a last cleaning step on all cases as they are read.
        Augmentation is applied only on read cases that are from the train set.
    """

    def __init__(self, cache_root, plane, label, split, preprocess, augment=None):
        """
        split: "train", "val", or "test" — filters which cached cases to use
        augment: an Augment instance for train, or None for val/test
        """
        self.plane = plane
        self.label = label
        self.preprocess = preprocess
        self.augment = augment

        # All cases of the selected split
        self.case_paths = []
        for f in sorted(Path(cache_root).glob("*.npz")):
            data = np.load(f, allow_pickle=True)
            if str(data["split"]) == split:
                self.case_paths.append(f)

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        data = np.load(self.case_paths[idx], allow_pickle=True)

        case = {
            "case_id": str(data["case_id"]),
            "labels": {
                "abnormal": int(data["label_abnormal"]),
                "acl": int(data["label_acl"]),
                "meniscus": int(data["label_meniscus"]),
            },
            "planes": {
                "axial": data["axial"],
                "coronal": data["coronal"],
                "sagittal": data["sagittal"],
            },
        }

        if self.augment is not None:
            case = self.augment.run(case)

        # Normalization is done last after every other preprocessing operation has already been applied
        case = self.preprocess.normalize_case(case)

        volume = case["planes"][self.plane]  # (S, 224, 224)
        image_tensor = torch.from_numpy(volume).unsqueeze(1).repeat(1, 3, 1, 1).float()

        radiomics = torch.from_numpy(data[f"radiomics_{self.plane}"]).float()  # (45,)
        label = torch.tensor(case["labels"][self.label], dtype=torch.float32)

        return image_tensor, radiomics, label

def load_case_metadata(cache_root):
    """Lightweight scan of the cache (no image data) to feed DataSplit.compute_pos_weights,
    which expects plain case dicts with labels/split/cv_fold, not a CacheDataset."""
    cases = []
    for f in sorted(Path(cache_root).glob("*.npz")):
        data = np.load(f, allow_pickle=True)
        cases.append({
            "case_id": str(data["case_id"]),
            "labels": {
                "abnormal": int(data["label_abnormal"]),
                "acl": int(data["label_acl"]),
                "meniscus": int(data["label_meniscus"]),
            },
            "split": str(data["split"]),
            "cv_fold": int(data["cv_fold"]),
        })
    return cases


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PlaneModel(nn.Module):
    def __init__(self, cnn_features_dim=CNN_FEATURE_DIM, radiomics_dim=RADIOMICS_DIM):
        super().__init__()

        torch.hub.set_dir("./weights")  # reuse the same cache you already downloaded into
        backbone = torch.hub.load("Warvito/radimagenet-models", "radimagenet_resnet50")
        backbone.fc = nn.Identity()
        self.cnn_backbone = backbone

        combined_dim = cnn_features_dim + radiomics_dim
        self.classifier_head = nn.Linear(combined_dim, 1)

    def forward(self, images, radiomics):
        cnn_out = self.cnn_backbone(images)  # (S, 2048, 7, 7) — raw conv feature map, head removed
        cnn_out = F.adaptive_avg_pool2d(cnn_out, (1, 1))  # (S, 2048, 1, 1) — global average pool, added back
        cnn_out = cnn_out.reshape(cnn_out.shape[0], -1)  # (S, 2048)
        pooled_cnn = cnn_out.max(dim=0).values  # (2048,) — bagging across slices

        combined = torch.cat([pooled_cnn, radiomics])  # (2048 + 45,)
        logit = self.classifier_head(combined.unsqueeze(0))
        return logit.squeeze()


# ---------------------------------------------------------------------------
# Prediction + metrics helpers
# ---------------------------------------------------------------------------

def generate_predictions(model, dataset, device):
    """
    Runs inference over an entire dataset (no augmentation, no shuffling)
    and returns one row per case: its case_id, predicted probability, and
    true label.
    """

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    model.eval()
    with torch.no_grad():
        for i, (images, radiomics, lbl) in enumerate(loader):
            images = images.squeeze(0).to(device)
            radiomics = radiomics.squeeze(0).to(device)
            logit = model(images, radiomics)
            prob = torch.sigmoid(logit).item()
            case_id = dataset.case_paths[i].stem  # filenames are "{case_id}.npz"
            results.append({"case_id": case_id, "prob": prob, "true_label": lbl.item()})
    return results

def metrics_from_predictions(predictions, threshold=0.5):
    """
    Builds a Metrics object (confusion-matrix counts + AUCs) from a list
    of {"case_id", "prob", "true_label"} dicts, e.g. generate_predictions()'s output.
    """

    y_true = np.array([r["true_label"] for r in predictions])
    y_scores = np.array([r["prob"] for r in predictions])
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    return Metrics(tp, fp, tn, fn, y_true=y_true, y_scores=y_scores)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(plane, label, epochs, lr):

    preprocess = PreProcess()
    augment = Augment()

    train_set = CacheDataset(CACHE_ROOT, plane, label, "train", preprocess, augment=augment)
    val_set = CacheDataset(CACHE_ROOT, plane, label, "val", preprocess, augment=None)

    print(f"[{plane}/{label}] Train cases: {len(train_set)}, Val cases: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    all_case_metadata = load_case_metadata(CACHE_ROOT)
    pos_weights = DataSplit.compute_pos_weights(all_case_metadata, split="train")
    pos_weight = torch.tensor(pos_weights[label], device=DEVICE)
    print(f"pos_weight for '{label}': {pos_weights[label]:.3f}")

    model = PlaneModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_name = f"best_{plane}_{label}_model.pth" # Save the best weights
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, radiomics, lbl in train_loader:
            images = images.squeeze(0).to(DEVICE)
            radiomics = radiomics.squeeze(0).to(DEVICE)
            lbl = lbl.squeeze(0).to(DEVICE)

            optimizer.zero_grad()
            logit = model(images, radiomics)
            loss = loss_fn(logit, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation — decides which epoch's weights to actually keep.
        val_preds = generate_predictions(model, val_set, DEVICE)
        val_metrics = metrics_from_predictions(val_preds)

        print(f"[{plane}/{label}] Epoch {epoch + 1}/{epochs} — train loss: {avg_loss:.4f} "
              f"— val AUC-ROC: {val_metrics.auc_roc:.4f} — val sensitivity: {val_metrics.sensitivity:.4f} "
              f"— val specificity: {val_metrics.specificity:.4f}")

        if val_metrics.auc_roc is not None and val_metrics.auc_roc > best_val_auc:
            best_val_auc = val_metrics.auc_roc
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  New best model saved: {checkpoint_name} (val AUC-ROC {best_val_auc:.4f})")

    print(f"[{plane}/{label}] Training done. Best val AUC-ROC: {best_val_auc:.4f}")

    # Reload the BEST checkpoint before generating final test predictions
    model.load_state_dict(torch.load(checkpoint_name, map_location=DEVICE))

    # Generate predictions on the held-out test split
    test_set = CacheDataset(CACHE_ROOT, plane, label, "test", preprocess, augment=None)
    test_preds = generate_predictions(model, test_set, DEVICE)

    # Save the predictions as a CSV
    pred_path = f"predictions_{plane}_{label}.csv"
    pd.DataFrame(test_preds).to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    return best_val_auc



if __name__ == "__main__":
    ...