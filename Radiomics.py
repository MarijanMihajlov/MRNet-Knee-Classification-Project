"""
Radiomics Feature Extractor

Extracts hand-crafted radiomics features from a single MRNet case.
Each case has three .npy files: axial.npy, coronal.npy, sagittal.npy
"""

import numpy as np
from scipy import stats, ndimage
from scipy.ndimage import gaussian_laplace
from skimage.feature import graycomatrix, graycoprops
import pywt


class RadiomicsExtractor:
    """
    Manual radiomics featyre extractor for pre-processed cases.

    Input:
        case = {
            "case_id": "0000",
            "planes": {
                "axial":    np.ndarray,  # (S, H, W), float32
                "coronal":  np.ndarray,  # (S, H, W), float32
                "sagittal": np.ndarray,  # (S, H, W), float32
            },
            "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
        }

    Output per plane: 45 features
        - 13 first-order stats
        - 6  GLCM texture stats
        - 13 LoG-filtered first-order stats
        - 13 wavelet-approximation first-order stats

    """

    def __init__(self):
        self._graycomatrix = graycomatrix
        self._graycoprops = graycoprops

    def extract(self, case: dict) -> dict:
        """Returns {'axial': arr, 'coronal': arr, 'sagittal': arr}."""
        features = {}
        for plane in ["axial", "coronal", "sagittal"]:
            vol = case["planes"][plane]
            mask = self._auto_mask(vol)
            features[plane] = self._extract_plane(vol, mask)
        return features

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_plane(self, volume, mask):
        region = volume[mask > 0]
        if region.size == 0:
            region = volume.flatten()

        feats = []

        # 1. First-order statistics
        feats.extend(self._first_order(region))

        # 2. GLCM texture
        feats.extend(self._glcm_features(volume, mask))

        # 3. LoG-filtered first-order
        feats.extend(self._log_features(volume, mask))

        # 4. Wavelet-approximation first-order
        feats.extend(self._wavelet_features(volume, mask))

        return np.array(feats, dtype=np.float32)

    def _auto_mask(self, volume):
        """Foreground mask: threshold above 5th percentile + largest blob."""
        mask = (volume > np.percentile(volume, 5)).astype(np.uint8)
        labeled, num = ndimage.label(mask)
        if num > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            largest = np.argmax(sizes) + 1
            mask = (labeled == largest).astype(np.uint8)
        return mask

    def _first_order(self, region):
        """13 first-order statistical features"""
        mean = float(np.mean(region))
        std = float(np.std(region))
        var = float(np.var(region))
        skew = float(stats.skew(region))
        kurt = float(stats.kurtosis(region))
        median = float(np.median(region))
        p10, p90 = np.percentile(region, [10, 90])
        iqr = float(np.percentile(region, 75) - np.percentile(region, 25))
        rng = float(np.max(region) - np.min(region))
        energy = float(np.sum(region ** 2))
        rms = float(np.sqrt(np.mean(region ** 2)))

        # Histogram entropy (log2)
        hist, _ = np.histogram(region, bins=32, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

        return [mean, std, var, skew, kurt, entropy, median, float(p10), float(p90), iqr, rng, energy, rms]

    def _glcm_features(self, volume, mask):
        """6 GLCM texture features averaged across slices and angles."""
        props = ["contrast", "dissimilarity", "homogeneity","energy", "correlation", "ASM"]
        slice_feats = []

        for i in range(volume.shape[0]):
            slc = volume[i]
            slc_mask = mask[i]

            # Crop to bounding box of mask to exclude background
            ys, xs = np.where(slc_mask > 0)
            if len(ys) < 2:
                continue

            ymin, ymax = ys.min(), ys.max() + 1
            xmin, xmax = xs.min(), xs.max() + 1
            crop = slc[ymin:ymax, xmin:xmax]
            crop_mask = slc_mask[ymin:ymax, xmin:xmax]

            roi_vals = crop[crop_mask > 0]
            if roi_vals.size < 2:
                continue

            vmin, vmax = roi_vals.min(), roi_vals.max()
            if vmax - vmin < 1e-6:
                continue

            # Discretize to 32 bins (0 is reserved for background)
            disc = np.zeros_like(crop, dtype=np.uint8)
            disc[crop_mask > 0] = (
                    np.clip((roi_vals - vmin) / (vmax - vmin) * (32 - 1),
                            0, 32 - 1).astype(np.uint8) + 1
            )

            glcm = self._graycomatrix(
                disc,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=32 + 1,
                symmetric=True,
                normed=True,
            )
            per_slice = [float(np.mean(self._graycoprops(glcm, p))) for p in props]
            slice_feats.append(per_slice)

        if not slice_feats:
            return [0.0] * len(props)

        return np.mean(slice_feats, axis=0).tolist()

    def _log_features(self, volume, mask):
        """First-order stats on Laplacian-of-Gaussian filtered volume."""
        filtered = gaussian_laplace(volume, sigma=1.0)
        region = filtered[mask > 0]
        if region.size == 0:
            region = filtered.flatten()
        return self._first_order(region)

    def _wavelet_features(self, volume, mask):
        """First-order stats on wavelet approximation coefficients, ROI-masked."""
        coeffs = []
        for i in range(volume.shape[0]):
            slc_mask = mask[i]
            if slc_mask.sum() == 0:
                continue

            # Zero out background so the DWT only "sees" the ROI
            masked_slice = np.where(slc_mask > 0, volume[i], 0.0)
            cA, _ = pywt.dwt2(masked_slice, "db1")

            # Downsample the mask to match cA's resolution (db1 halves each dim)
            cA_mask = slc_mask[::2, ::2][: cA.shape[0], : cA.shape[1]]
            coeffs.append(cA[cA_mask > 0])

        if not coeffs:
            return [0.0] * 13

        all_coeffs = np.concatenate([c.flatten() for c in coeffs if c.size > 0])
        if all_coeffs.size == 0:
            return [0.0] * 13

        return self._first_order(all_coeffs)
