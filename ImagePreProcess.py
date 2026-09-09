import SimpleITK as sitk
import numpy as np
import torch
from torchvision.transforms import functional as F

"""
        Raw .npy
            ↓
        N4 bias correction
            ↓
        Histogram standardization  ← fit on train, transform all
            ↓
        Crop ROI
            ↓
        Resize to fixed size
            ↓
        Per-slice zero-mean / unit-variance
"""


"""Provides all of the functions for cleaning the dataset"""
class PreProcess:

    def __init__(self):
        self.histogram_standard_scale = None
        self.histogram_landmarks = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    def fit(self, train_cases_raw):
        """Call once, on train cases only, before running the full pipeline on anyone."""
        n4_train = [self.n4_field_correction(c) for c in train_cases_raw]
        n4_train = [self.crop_region_of_interest(c) for c in n4_train]  # see point 2
        self.fit_histogram_standardization(n4_train)


    def normalize_case(self,case, unit_variance = True):

        """
        Applies per-slice zero-mean + unit-varience normalization to every slice.

        Expected Input:
            {
                "case_id": "0000",
                "planes":{
                    "axial":    np.ndarray,   # (S, H, W), float32, per-slice standardized
                    "coronal":  np.ndarray,   # (S, H, W), float32, per-slice standardized
                    "sagittal": np.ndarray,   # (S, H, W), float32, per-slice standardized
                    },
                "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
            }

        Returns a dict with the same structure.

        Arguments:
            case: Dict with 'case_id', 'planes' and 'labels'
            unit_variance: Boolean to control wether to apply unit-varience normalization

        """

        normalized = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict(),
        }

        for plane, image_array in case["planes"].items():
            out = np.empty_like(image_array)

            for i in range(image_array.shape[0]):
                slc = image_array[i]
                centered = slc - np.mean(slc)

                if unit_variance:
                    std = np.std(slc)
                    if std > 1e-8:
                        centered = centered / std

                out[i] = centered

            normalized["planes"][plane] = out

        return normalized


    def n4_field_correction(self, case):

        """
        Applies N4 bias field correction to each plane of a case.

        Expected Input:
        {
            "case_id": "0000",
            "paths": {
                "axial": Path(".../train/axial/0000.npy"),
                "coronal": Path(".../train/coronal/0000.npy"),
                "sagittal": Path(".../train/sagittal/0000.npy"),
            },
            "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
        }

        Returns:
        dict with simillar structure:
        {
            "case_id": str,
            "labels":  dict,
            "planes": {
                "axial":    np.ndarray of shape (S, H, W), float32, N4-corrected,
                "coronal":  np.ndarray of shape (S, H, W), float32, N4-corrected,
                "sagittal": np.ndarray of shape (S, H, W), float32, N4-corrected,
            },
        }

        Note:
            This is computationally expensive (~30–120 s per 3D volume).
            For training pipelines, pre-process and cache the output to disk
            rather than running it inside the data-loading loop.

        """

        normalized = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict(),
        }

        for plane, path in case["paths"].items():

            vol = np.load(path).astype(np.float32)

            # Convert numpy array to SimpleITK image
            image = sitk.GetImageFromArray(vol)

            # Run N4
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected = corrector.Execute( sitk.Cast(image,sitk.sitkFloat32) )

            # Convert back to numpy
            normalized_image_array = sitk.GetArrayFromImage(corrected)

            normalized["planes"][plane] = normalized_image_array

        return normalized


    def crop_region_of_interest(self, case, pad = 10):

        cropped = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict()
        }

        for plane, image_array in case["planes"].items():

            # Collapse across slices to find any signal in the volume
            mask = (image_array > np.percentile(image_array,1)).astype(np.uint8) # bottom 1% as bg

            # Find bounding box across all slices
            rows = np.where( np.any(mask, axis=(0,2)) )[0]
            cols = np.where( np.any(mask, axis=(0,1)) )[0]

            # Fallback
            if len(rows) == 0 or len(cols) == 0:
                cropped["planes"][plane] = image_array
                continue

            r_min, r_max = max(0, rows[0] - pad), min(image_array.shape[1], rows[-1] + pad)
            c_min, c_max = max(0, cols[0] - pad), min(image_array.shape[2], cols[-1] + pad)

            cropped["planes"][plane] = image_array[:, r_min:r_max, c_min:c_max]

        return cropped


    def resize_planes(self, case, target_size=(224,224) ):

        """
                Resize all planes to a consistent spatial resolution.

                Required after ROI cropping because bounding boxes vary per patient,
                producing arrays of different (H, W). CNNs need fixed input dimensions
                to batch multiple cases together.

                Expected Input:
                    {
                        "case_id": str,
                        "labels": dict,
                        "planes": {
                            "axial":    np.ndarray of shape (S, H, W), float32,
                            "coronal":  np.ndarray of shape (S, H, W), float32,
                            "sagittal": np.ndarray of shape (S, H, W), float32,
                        },
                    }

                Returns:
                    Same structure with arrays resized to
                    (S, target_size[0], target_size[1]).

                Note:
                    Uses bilinear interpolation with antialiasing, which is
                    appropriate for intensity images. Pass a single int (e.g., 224)
                    for square output, or a tuple (H, W) for rectangular.
        """

        resized = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict(),
        }

        for plane, image_array in case["planes"].items():

            # (S, H, W) -> (S, 1, H, W) so torchvision treats S as batch
            tensor = torch.from_numpy(image_array).unsqueeze(1)

            # Bilinear resize with antialiasing
            resized_tensor = F.resize(
                tensor,
                size=target_size,
                antialias=True,
            )

            # (S, 1, H, W) -> (S, H, W) back to numpy
            resized["planes"][plane] = resized_tensor.squeeze(1).numpy().astype(np.float32)

        return resized


    def fit_histogram_standardization(self, train_cases, mask_percentile = 1.0):

        """
            Fit Nyul's histogram standardization on training data only.
            Stores the standard scale to be used later for transform.

            This should be called once on your training split BEFORE
            transforming validation and test data.

            Args:
                train_cases: List of case dicts with 'planes' -> np.ndarray
                mask_percentile: Bottom percentile treated as background
                                 and excluded from histogram fitting.
        """

        self.histogram_standard_scale = {}

        for plane in ["axial", "coronal", "sagittal"]:
            plane_landmarks = []

            for case in train_cases:
                image_array = case["planes"][plane]

                # Exclude background (near-zero voxels) so they don't dominate the percentile calculation
                threshold = np.percentile(image_array, mask_percentile)
                foreground = image_array[image_array>threshold]
                if foreground.size == 0:
                    continue

                # Record intensity values at each landmark percentile
                vals = np.percentile(foreground, self.histogram_landmarks)
                plane_landmarks.append(vals)

            # Standard scale = median across training set
            standard_scale = np.median(np.stack(plane_landmarks), axis=0)
            self.histogram_standard_scale[plane] = standard_scale


    def transform_histogram_standardization(self, case):

        """
            Apply fitted histogram standardization to a single case.
            Can be called on training, validation, or test cases.

            Expected Input:
                {
                    "case_id": str,
                    "labels": dict,
                    "planes": {
                        "axial":    np.ndarray (S, H, W),
                        "coronal":  np.ndarray (S, H, W),
                        "sagittal": np.ndarray (S, H, W),
                    },
                }

            Returns:
                Same structure with intensity-standardized arrays.
        """

        if self.histogram_standard_scale is None:
            raise RuntimeError(
                "Call fit_histogram_standardization() on training data first."
            )

        transformed = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict(),
        }

        for plane, image_array in case["planes"].items():

            # Compute this image's own landmarks
            threshold = np.percentile(image_array, 1.0)
            foreground = image_array[image_array > threshold]
            if foreground.size == 0:
                transformed["planes"][plane] = image_array
                continue

            source_scale = np.percentile(foreground, np.array(self.histogram_landmarks))
            target_scale = self.histogram_standard_scale[plane]

            # Piecewise linear mapping: source_scale -> target_scale
            transformed["planes"][plane] = self._nyul_piecewise_linear(
                image_array, source_scale, target_scale
            )

        return transformed

    @staticmethod
    def _nyul_piecewise_linear(image, source_scale, target_scale):
        """
        Piecewise linear interpolation mapping.
        Intensities between landmarks are linearly interpolated;
        intensities outside are linearly extrapolated.
        """
        flat = image.ravel().astype(np.float32)
        out = np.empty_like(flat)

        s = source_scale
        t = target_scale

        # Slopes for extrapolation below/above landmark range
        if len(s) > 1:
            slope_low = (t[1] - t[0]) / (s[1] - s[0] + 1e-8)
            slope_high = (t[-1] - t[-2]) / (s[-1] - s[-2] + 1e-8)
        else:
            slope_low = slope_high = 1.0

        # Below first landmark
        mask_low = flat < s[0]
        out[mask_low] = t[0] + slope_low * (flat[mask_low] - s[0])

        # Above last landmark
        mask_high = flat > s[-1]
        out[mask_high] = t[-1] + slope_high * (flat[mask_high] - s[-1])

        # Between landmarks (piecewise linear)
        mask_mid = ~(mask_low | mask_high)
        if np.any(mask_mid):
            out[mask_mid] = np.interp(flat[mask_mid], s, t)

        return out.reshape(image.shape)