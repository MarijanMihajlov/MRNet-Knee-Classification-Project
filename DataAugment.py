import numpy as np
import torch
from torchvision.transforms import functional as F
from scipy.ndimage import gaussian_filter, map_coordinates, gaussian_filter1d

"""
Augmentation is applied AFTER PreProcess.clean() and BEFORE normalize_case().
Only ever called on TRAIN-split cases -- val/test never pass through this class.

    Cached, deterministic (PreProcess.clean):
        N4 -> crop ROI -> histogram standardize -> resize
                    |
                    v   (train split only, fresh random draw every epoch)
              Augment.run()
                    |
                    v
        PreProcess.normalize_case()   <- always last, for every split

No fit/transform split needed here: every method takes randomly sampled
parameters and applies them to whatever case it's given. There's nothing
learned from the dataset, so there's no leakage risk inside this class --
the only thing that matters is that it's never called on val/test cases,
which is enforced by the dataloader, not by this class.
"""


class Augment:

    def __init__(
            self,
            rotation_range=15.0,  # degrees, +/-
            translate_frac=0.03,  # fraction of H/W, +/-
            scale_range=(0.95, 1.05),
            elastic_alpha=8.0,  # displacement magnitude, pixels
            elastic_sigma=6.0,  # smoothness of the displacement field
            noise_std=0.02,  # relative to image std
            blur_sigma_range=(0.0, 0.6),
            intensity_jitter=0.1,  # brightness/contrast fraction
            p_spatial=0.8,  # probability of applying spatial aug at all
            p_elastic=0.3,
            p_noise=0.3,
            p_blur=0.2,
            p_intensity=0.5,
            seed=None,
    ):
        self.rotation_range = rotation_range
        self.translate_frac = translate_frac
        self.scale_range = scale_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.noise_std = noise_std
        self.blur_sigma_range = blur_sigma_range
        self.intensity_jitter = intensity_jitter
        self.p_spatial = p_spatial
        self.p_elastic = p_elastic
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_intensity = p_intensity
        self.rng = np.random.default_rng(seed)

    def run(self, case):
        """
        Applies one random parameter draw per plane, identically to every
        slice in that plane's volume, so anatomy stays continuous slice
        to slice.

        Expected input / output: same case dict structure as PreProcess
        ("case_id", "labels", "planes": {axial, coronal, sagittal}).
        """
        augmented = {
            "case_id": case["case_id"],
            "labels": case["labels"],
            "planes": dict(),
        }

        for plane, volume in case["planes"].items():
            augmented["planes"][plane] = self._augment_volume(volume)

        return augmented

    # ------------------------------------------------------------------
    # Per-volume augmentation
    # -----------------------------------------------------------------
    def _augment_volume(self,volume):
        out = volume.copy()

        if self.rng.random() < self.p_spatial:
            angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
            tx = self.rng.uniform(-self.translate_frac, self.translate_frac) * out.shape[2]
            ty = self.rng.uniform(-self.translate_frac, self.translate_frac) * out.shape[1]
            scale = self.rng.uniform(*self.scale_range)
            out = self._apply_affine(out, angle, tx, ty, scale)

        if self.rng.random() < self.p_elastic:
            out = self._apply_elastic(out, self.elastic_alpha, self.elastic_sigma)

        if self.rng.random() < self.p_intensity:
            brightness = self.rng.uniform(-self.intensity_jitter, self.intensity_jitter)
            contrast = self.rng.uniform(1 - self.intensity_jitter, 1 + self.intensity_jitter)
            out = self._apply_intensity_jitter(out, brightness, contrast)

        if self.rng.random() < self.p_blur:
            sigma = self.rng.uniform(*self.blur_sigma_range)
            out = self._apply_blur(out, sigma)

            # noise is the one exception -- independent per slice is fine,
            # since real scanner noise isn't spatially correlated across slices
        if self.rng.random() < self.p_noise:
            out = self._apply_noise(out, self.noise_std)

        return out


    # ------------------------------------------------------------------
    # Individual transforms -- all operate on a full (S, H, W) volume
    # ------------------------------------------------------------------

    def _apply_affine(self, volume, angle, tx, ty, scale):
        tensor = torch.from_numpy(volume).unsqueeze(1) # (S,1,H,W)
        out = F.affine(tensor,angle=angle,translate=[tx,ty], scale=scale, shear=[0.0,0.0])
        return out.squeeze(1).numpy().astype(np.float32) # (S,H,W)

    def _apply_elastic(self, volume, alpha, sigma):

        h,w = volume.shape[1], volume.shape[2]
        dx = gaussian_filter(self.rng.uniform(-1, 1, (h, w)), sigma) * alpha
        dy = gaussian_filter(self.rng.uniform(-1, 1, (h, w)), sigma) * alpha

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]

        out = np.empty_like(volume)
        for i in range(volume.shape[0]):
            out[i] = map_coordinates(volume[i], coords, order=1, mode="reflect")

        return out.astype(np.float32)

    def _apply_intensity_jitter(self, volume, brightness, contrast):
        mean = volume.mean()
        return ((volume - mean) * contrast + mean + brightness * volume.std()).astype(np.float32)

    def _apply_blur(self, volume, sigma):
        if sigma <= 0:
            return volume
        # same sigma across slices for consistency; blur is 2D per slice
        out = np.empty_like(volume)
        for i in range(volume.shape[0]):
            out[i] = gaussian_filter(volume[i], sigma)
        return out.astype(np.float32)

    def _apply_noise(self, volume, relative_std):
        std = volume.std() * relative_std
        noise = self.rng.normal(0, std, size=volume.shape)
        return (volume + noise).astype(np.float32)
