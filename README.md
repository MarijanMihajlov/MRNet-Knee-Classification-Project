# MRNet Knee Injury Classification

A deep learning + radiomics ensemble pipeline for classifying knee MRI scans from the Stanford MRNet dataset across three diagnostic labels: **abnormal**, **ACL tear**, and **meniscus tear**.

## Table of Contents

- [Dataset](#dataset)
- [Data Loading](#data-loading)
- [Data Auditing](#data-auditing)
- [Image Pre-Processing](#image-pre-processing)
- [Data Augmentation](#data-augmentation)
- [Dataset Splitting and Balance Strategy](#dataset-splitting-and-balance-strategy)
- [CNN Architecture and Transfer Learning](#cnn-architecture-and-transfer-learning)
- [Feature Extraction and Clustering](#feature-extraction-and-clustering)
- [Radiomics](#radiomics)
- [Ensemble Learning and Classification](#ensemble-learning-and-classification)
- [Classification Task](#classification-task)
- [Evaluation Metrics](#evaluation-metrics)

---

## Dataset

The data used for training and inference comes from the publicly available **MRNet** dataset released by Stanford. It contains knee MRI scans from patients, with each scan performed in three planes: **axial**, **coronal**, and **sagittal**. Each plane contains roughly 40–60 slice images of the knee from that viewing angle.

Every scan carries three binary labels:

- **abnormal** — whether the knee shows any abnormality at all
- **acl** — whether an ACL injury is present
- **meniscus** — whether a meniscus injury is present

These labels follow a logical constraint: if a case is not abnormal, it cannot have an ACL or meniscus injury (`abnormal = 0` ⇒ `acl = 0` and `meniscus = 0`). Equivalently, if either an ACL or meniscus injury is present, the case must be labeled abnormal (`acl = 1` or `meniscus = 1` ⇒ `abnormal = 1`). Abnormal cases are not limited to ACL or meniscus injuries — the label also covers other knee abnormalities.

Looking at any individual slice, the knee appears cropped into a bounding box near the center of the image, surrounded by black background. All scans originate from the same facility, so the scanner hardware is fixed and intensity values are reasonably predictable across the dataset — this would not hold if scans were pooled from multiple institutions or scanner models.

The official MRNet release provides train, validation, and test splits, but the test set's labels are not public. Since only the labeled **train** and **validation** sets can be used, they are combined (~1,248 usable cases, after excluding a small number with missing plane files) and re-split into a new stratified train/val/test partition for this project (see [Dataset Splitting and Balance Strategy](#dataset-splitting-and-balance-strategy)).

## Data Loading

The dataset is initially loaded into a list of dictionaries, one per case, with the following structure:

```python
{
    "case_id": case_id,
    "orig_split": split,        # "train" or "valid" — which official folder the case came from
    "paths": paths,             # {"axial": Path(...), "coronal": Path(...), "sagittal": Path(...)}
    "labels": {
        "abnormal": int(row["abnormal"]),
        "acl": int(row["acl"]),
        "meniscus": int(row["meniscus"]),
    },
}
```

## Data Auditing

Before any processing, every case is validated against the following checks:

- **File existence** — the `.npy` file for each plane actually exists on disk
- **Path validity** — the filename (e.g. `0001.npy`) matches the case's expected ID
- **Plane completeness** — the case has all three expected planes (axial, coronal, sagittal)
- **Label check** — logical constraints hold, values are binary, and all expected label keys are present
- **Image array integrity** — each `.npy` array is checked for correct shape, data type, emptiness, and invalid values (e.g. NaNs)

## Image Pre-Processing

Pre-processing steps are separated into two categories based on data-leakage risk.

**Safe from data leakage** (can be applied independently, per case, with no dependence on the rest of the dataset):

- **Per-image normalization** — each image is rescaled to zero-mean/unit-variance (or min-max to [0, 1]) individually, never using a global min/max computed across the dataset. MRI intensity scales aren't comparable across scans the way pixel values are across ordinary photos, so per-image normalization is required at minimum.
- **N4 bias field correction** — corrects intensity inhomogeneity, a known MRI artifact caused by smooth shading from the scanner coil, unrelated to anatomy. This is a standard MRI-specific correction (available via SimpleITK/ANTsPy) and matters more here than any natural-image preprocessing step would.
- **Resampling to consistent voxel spacing** — not needed for this project, since the MRNet dataset is already pre-processed to consistent resolution.
- **Cropping to a region of interest** — knee MRIs contain substantial irrelevant background. Cropping to a consistent knee-centered bounding box reduces the input the model has to process and removes a source of nuisance variation.

**Not safe from data leakage** (must be fit on training data only, then applied to all splits):

- **Histogram standardization / intensity matching** (Nyúl's method) — maps every scan's intensity histogram onto a common reference distribution, reducing scanner-to-scanner variation. Since fitting this transform requires seeing the population's intensity distribution, it is fit on the training split only and then applied identically to validation and test data, avoiding leakage of test-set statistics into training.

## Data Augmentation

Applied only to the training split, with a fresh random draw every epoch, augmentation modifies each training sample (rotation, crop, noise, intensity, scaling) so the model learns to treat these variations as irrelevant to the underlying anatomy. Augmentations are chosen to remain anatomically plausible:

- Small-angle rotation (±10–15°)
- Small translation/crop jitter (a few percent of image size)
- Small scale/zoom jitter, simulating field-of-view variation
- Elastic deformation — a well-validated, medical-imaging-specific augmentation mimicking natural soft-tissue variability
- Gaussian noise / slight Gaussian blur, simulating scanner noise
- Intensity/contrast jitter — small random brightness/contrast shifts, helping the model generalize across scanner differences

## Dataset Splitting and Balance Strategy

The combined dataset is split into train/val/test sets, stratified jointly on all three labels, with additional stratified cross-validation folds computed within the training split. This preserves the original class ratio for each label across every split and fold.

To further address class imbalance, positive-class weights are computed from the training split and applied during loss calculation, penalizing misclassification of the minority class (e.g. an injury) more heavily than the majority class.

## CNN Architecture and Transfer Learning

Each slice is processed through a **ResNet-50** architecture — a 50-layer deep convolutional network — initialized with **RadImageNet** pretrained weights. RadImageNet is a large-scale medical imaging dataset spanning CT, MRI, and ultrasound modalities, making it a more relevant starting point than natural-image pretraining (e.g. ImageNet) for this task.

ResNet-50 expects a 4D float tensor of normalized 3-channel images:

```
Shape:  (B, C, H, W)
        B — batch size (number of images processed at once)
        C — channels (must be exactly 3, matching RGB)
        H — height (224 by default)
        W — width  (224 by default)

Pixels: raw pixel values (0–255) scaled to the float range [0.0, 1.0]

RadImageNet normalization constants (per channel):
        Mean:                [0.485, 0.456, 0.406]
        Standard deviation:   [0.229, 0.224, 0.225]
```

Since MRI slices are single-channel (grayscale), the single channel is repeated three times to satisfy the 3-channel input requirement.

The backbone is used in two distinct modes across the project:

- **Feature extraction / clustering** — the backbone runs **frozen**, exactly as pretrained on RadImageNet, with no fine-tuning. This provides a neutral, unbiased lens for auditing the dataset before any task-specific learning occurs.
- **Classification / ensemble** — nine separate copies of the backbone (one per plane × label combination) are fine-tuned end-to-end on the knee MRI data, each starting from the same RadImageNet initialization but diverging as training proceeds.

## Feature Extraction and Clustering

For feature extraction, the network's final classification layer is replaced with an identity layer (`f(x) = x`), so the model outputs its pooled internal features directly instead of class probabilities. For a single plane's volume, this produces an array of shape `(S, 2048)`, where `S` is that plane's slice count — each case yields three such arrays, one per plane. Computation runs on GPU (CUDA) when available, falling back to CPU otherwise.

For clustering, these feature vectors are first reduced in dimensionality via PCA, then passed to a K-means algorithm. The resulting clusters are visualized on a 2D plot (via t-SNE/UMAP) to audit the dataset for structure, outliers, or unexpected groupings before any classifier is trained.

## Radiomics

Hand-crafted features are combined with the CNN's learned features — a well-established approach in medical imaging known as **radiomics**. Rather than concatenating at the individual slice level, radiomics features are computed once per plane volume and concatenated with the CNN's pooled, slice-aggregated feature vector before the final classifier head.

Radiomics features computed per plane include:

- First-order statistics (mean, variance, skewness, kurtosis, entropy, and related measures of the pixel intensity distribution)
- Texture features (GLCM — gray-level co-occurrence matrix)
- Laplacian-of-Gaussian and wavelet-filtered versions of the above first-order statistics

In principle, radiomics features are most meaningful when computed within a specific region of interest (e.g. just the meniscus, rather than the whole knee); for this project they are computed over an automatically thresholded foreground mask of the whole knee volume.

## Ensemble Learning and Classification

The final classifier combines three distinct ensembling ideas, layered on top of each other:

**1. Bagging across slices.** Each plane's volume contains a variable number of slices. Every slice is passed independently through the CNN backbone to produce a per-slice feature vector, and these are combined via **max-pooling** across the slice dimension into a single fixed-length vector for the whole volume — the injury-relevant signal is typically concentrated in only a few slices, so max-pooling lets the strongest evidence dominate rather than being diluted by an average across many uninformative slices.

**2. Ensembling across anatomical planes.** Axial, coronal, and sagittal each have their own independently trained CNN, since different planes reveal different pathology (e.g. sagittal views are typically most informative for ACL tears). Each plane's pooled CNN feature vector is concatenated with that plane's radiomics feature vector, then passed through a linear classifier head, producing one injury probability per plane.

**3. Hand-crafted + learned feature fusion.** Within each plane's classifier head, CNN-learned features and hand-crafted radiomics features are fused by concatenation before the final linear layer, letting the classifier draw on both learned visual patterns and explicit statistical/textural descriptors.

**Combining the three planes.** The three planes' probabilities are combined via **simple averaging** to produce the final per-label prediction. A learned meta-learner (logistic regression stacking on the three plane probabilities) was considered and prototyped, but simple averaging was chosen instead as the final approach — it requires no additional fitting step, avoids the risk of overfitting a meta-learner on a relatively small validation set, and was judged to offer the best trade-off between complexity and expected benefit given the project's scope and timeline.

**Training and model selection.** Each of the nine plane/label combinations is trained independently, with validation performance (AUC-ROC, via the project's `Metrics` module) monitored after every epoch. The checkpoint with the best validation AUC-ROC is retained — not simply the model state after the last epoch — guarding against overfitting in later epochs before any test-set evaluation occurs.

## Classification Task

For each case in the held-out test set, the ensemble outputs three separate probabilities — one each for **abnormal**, **acl**, and **meniscus** — each derived independently from its own three-plane ensemble, as described above.

## Evaluation Metrics

Model performance is evaluated using the following metrics, computed on the held-out test set for both each individual plane and the final averaged ensemble:

- Sensitivity / Recall
- Specificity
- Precision
- Negative Predictive Value (NPV)
- Accuracy
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- F-beta scores (F1, F2)
- AUC-ROC / AUC-PR
