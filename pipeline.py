import numpy as np
import pandas as pd
from pathlib import Path

from DataAudit import DataAudit
from DataSplit import DataSplit
from ImagePreProcess import PreProcess
from Radiomics import RadiomicsExtractor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = "./MRNet-dataset"
FILES_DIR = Path("./MRNet-dataset/Files")
TABLES_DIR = "./MRNet-dataset/Tables"
CACHE_ROOT = Path("./Cache")

train_abnormal_path = "./MRNet-dataset/Tables/train_abnormal.csv"
train_acl_path = "./MRNet-dataset/Tables/train_acl.csv"
train_meniscus_path = "./MRNet-dataset/Tables/train_meniscus.csv"
valid_abnormal_path = "./MRNet-dataset/Tables/valid_abnormal.csv"
valid_meniscus_path = "./MRNet-dataset/Tables/valid_meniscus.csv"
valid_acl_path = "./MRNet-dataset/Tables/valid_acl.csv"

csv_paths = {

    "train":{
        "abnormal": {
                "path": train_abnormal_path,
            },

        "acl":{
            "path": train_acl_path,
        },

        "meniscus":{
            "path": train_meniscus_path,
        },
    },

    "valid":{
        "abnormal": {
            "path": valid_abnormal_path,
        },

        "acl": {
            "path": valid_acl_path,
        },

        "meniscus": {
            "path": valid_meniscus_path,
        },
    },

}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def atomic_save(final_path, **kwargs):
    """
    Saving file as an atomic operation.
    Writes to a temp file first, then renames to the final path.
    os.replace() is atomic, so the final path only ever exists in a
    complete, valid state"""
    import os
    final_path = Path(final_path)
    tmp_path = final_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **kwargs)
    os.replace(tmp_path, final_path)

def load_data():
    """
        The returned dataset has the following structure:

        {
            "case_id": "0000",
            "paths": {
                "axial": Path(".../train/axial/0000.npy"),
                "coronal": Path(".../train/coronal/0000.npy"),
                "sagittal": Path(".../train/sagittal/0000.npy"),
            },
            "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
        }
    """
    dataset = []

    # Functoin to load the labels
    def load_labels(csv_path, label_name):
        df = pd.read_csv(csv_path, header=0, names=["case_id", label_name])
        df["case_id"] = df["case_id"].astype(str).str.zfill(4)  # "0" -> "0000"
        return df

    # Load the labels and combine them
    split = []
    for split_name, tables in csv_paths.items():
        abnormal = load_labels(tables["abnormal"]["path"], "abnormal")
        acl = load_labels(tables["acl"]["path"], "acl")
        meniscus = load_labels(tables["meniscus"]["path"], "meniscus")


        merged = abnormal.merge(acl, on="case_id").merge(meniscus, on="case_id")
        merged["orig_split"] = split_name  # keep track of where it came from
        split.append(merged)

    labels = pd.concat(split,ignore_index=True)

    # Combine the case_id, file paths and labels for each scan in one dictonary
    for _, row in labels.iterrows():
        case_id = row["case_id"]
        split = row["orig_split"] # "train" or "valid" — tells you which folder
        paths = {
            plane: FILES_DIR / split / plane / f"{case_id}.npy"
            for plane in ["axial", "coronal", "sagittal"]
        }
        missing = [p for p, path in paths.items() if not path.exists()]
        if missing:
            print(f"Warning: case {case_id} ({split}) missing planes {missing}")
            continue

        dataset.append({
            "case_id": case_id,
            "orig_split": split,
            "paths": paths,
            "labels": {
                "abnormal": int(row["abnormal"]),
                "acl": int(row["acl"]),
                "meniscus": int(row["meniscus"]),
            },
        })

    return dataset



if __name__ == "__main__":

    # Load the dataset
    dataset = load_data()

    # ------------------------------------------------------------------
    # Stratified train/val/test + CV fold split
    # ------------------------------------------------------------------

    # Save the split after the FIRST run and reuse on every subsequent run.
    SPLITS_PATH = Path("./splits.csv")

    if SPLITS_PATH.exists():
        print(f"Loading existing split assignment from {SPLITS_PATH} ...")
        splits_df = pd.read_csv(SPLITS_PATH, dtype={"case_id": str})
        split_of = dict(zip(splits_df["case_id"], splits_df["split"]))
        fold_of = dict(zip(splits_df["case_id"], splits_df["cv_fold"]))

        # Alert if missing cases exist
        missing = [c["case_id"] for c in dataset if c["case_id"] not in split_of]
        if missing:
            raise RuntimeError(
                f"{len(missing)} cases in your dataset aren't in {SPLITS_PATH}."
                f"Your dataset has changed since the split was created — delete {SPLITS_PATH} to regenerate it."
                f"Note this may reassign splits for cases already in the cache."
            )

        # Assign split and fold key/value to every case dict in the dataset
        for c in dataset:
            c["split"] = split_of[c["case_id"]]
            c["cv_fold"] = fold_of[c["case_id"]]
    else:
        print(f"No {SPLITS_PATH} found - computing split for the first time...")
        splitter = DataSplit(dataset)
        dataset = splitter.split_dataset()
        splitter.report(dataset) # Print split report (Optional)

        # Save the split assignment as CSV for future use
        pd.DataFrame([
            {"case_id": c["case_id"], "split": c["split"], "cv_fold": c["cv_fold"]}
            for c in dataset
        ]).to_csv(SPLITS_PATH, index=False)
        print(f"Saved split assignment to {SPLITS_PATH} for all future runs to reuse.")

    preprocess = PreProcess()
    radiomics = RadiomicsExtractor()

    # Cache paths for fully preprocessed cases and cases with N4 applied
    N4_CACHE_ROOT = Path("./Cache_n4cropped")
    N4_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Check if every case is preprocessed and cached
    all_case_ids = {c["case_id"] for c in dataset}
    cached_case_ids = {f.stem for f in CACHE_ROOT.glob("*.npz")}
    cache_is_complete = all_case_ids.issubset(cached_case_ids)

    if cache_is_complete:
        """
        If every case is already fully cached, Phase 1/2/histogram-fit
         would run for nothing so skip straight to training.
        """
        print(f"All {len(all_case_ids)} cases already cached in {CACHE_ROOT}.")
        print("Skipping Phase 1/2 and histogram fitting entirely.")


    if not cache_is_complete:

        # Set too false to skip N4 bias field correction (crop, histogram standardization, resize, normalize still runs).
        """
         NOTE: if you already have SOME cases cached with N4 applied and
         switch this to False, you'll get a dataset with inconsistent
         preprocessing (some N4'd, some not). For consistency, delete
         Cache_n4cropped/ and Cache/ before switching this flag.
        """
        ENABLE_N4 = False


        # Guard against silently mixing N4'd and non-N4'd cases in the same cache.
        # Writes a marker file recording which mode built this cache.
        n4_mode_marker = N4_CACHE_ROOT / "_n4_mode.txt"
        if n4_mode_marker.exists():
            previous_mode = n4_mode_marker.read_text().strip()
            current_mode = str(ENABLE_N4)
            if previous_mode != current_mode:
                raise RuntimeError(
                    f"Cache_n4cropped/ was built with ENABLE_N4={previous_mode}, "
                    f"but ENABLE_N4={current_mode} now. Mixing the two would give "
                    f"you an inconsistently preprocessed dataset. Delete "
                    f"Cache_n4cropped/ and Cache/ (and splits.csv, for a clean "
                    f"slate) before switching modes."
                )
        else:
            n4_mode_marker.write_text(str(ENABLE_N4))

        def load_raw_planes(case):
            """Returns a dict for the case with the raw .npy files loaded as float32 planes."""
            loaded = {
                "case_id": case["case_id"],
                "labels": case["labels"],
                "planes": dict(),
            }
            for plane, path in case["paths"].items():
                loaded["planes"][plane] = np.load(path).astype(np.float32)
            return loaded

        # ------------------------------------------------------------------
        # PHASE 1: N4 correction (if enabled) + crop. Runs ONCE per case, cached to disk.
        # ------------------------------------------------------------------

        step_label = "N4 correction + crop" if ENABLE_N4 else "crop (N4 disabled)"
        print(f"{step_label} starting...")

        for i, case in enumerate(dataset):
            n4_path = N4_CACHE_ROOT / f"{case['case_id']}.npz"
            if n4_path.exists():
                continue

            print(f"[{step_label} {i + 1}/{len(dataset)}] Processing case {case['case_id']}...")

            # N4 correction and crop / Only crop region of interest
            if ENABLE_N4:
                n4_case = preprocess.n4_field_correction(case)
            else:
                n4_case = load_raw_planes(case)

            n4_case = preprocess.crop_region_of_interest(n4_case)

            # Save the N4 + cropped file (or only cropped) in the N4 cache
            atomic_save(
                n4_path,
                case_id=n4_case["case_id"],
                axial=n4_case["planes"]["axial"],
                coronal=n4_case["planes"]["coronal"],
                sagittal=n4_case["planes"]["sagittal"],
            )

        print(f"{step_label} finished.")



        # ------------------------------------------------------------------
        # Fit histogram standardization directly on the CACHED N4+crop output..
        # ------------------------------------------------------------------

        def load_n4_cropped(case):
            """Loads a case with its full resolution data from disk, from the N4 cache."""
            data = np.load(N4_CACHE_ROOT / f"{case['case_id']}.npz", allow_pickle=True)
            return {
                "case_id": case["case_id"],
                "labels": case["labels"],
                "planes": {
                    "axial": data["axial"],
                    "coronal": data["coronal"],
                    "sagittal": data["sagittal"],
                },
            }
        class LazyCaseLoader:
            """
            Re-iterable wrapper that reloads each case from disk on every
            pass, instead of holding all cases' full-resolution volumes in
            memory at once.

            Instead of loading all cases into RAM simultaneously, this wrapper
            ensures only one case exists in memory at a time during processing.

            Needed because fit_histogram_standardization() loops over its input three times (once per plane) — a plain
            generator would be exhausted after the first pass.

            This keeps memory bounded to one case at a time, at the cost of re-reading each
            case's .npz from disk 3x.
            """

            def __init__(self, cases, loader_fn):
                self.cases = cases
                self.loader_fn = loader_fn # Function that takes a single case identifier and loads its full resolution data from disk.

            def __iter__(self):
                """
                Runs the loader_fn(c) on each case in real time as the loop requests it, yielding one
                loaded volume at a time and discarding it from memory when the loop moves to the next iteration.
                """
                for c in self.cases:
                    yield self.loader_fn(c)

        print("Fitting n4 historgram ...")
        train_cases = [c for c in dataset if c["split"] == "train"]
        train_n4_cropped = LazyCaseLoader(train_cases, load_n4_cropped)
        preprocess.fit_histogram_standardization(train_n4_cropped)
        print("Finished n4 histogram fitting.")


        # ------------------------------------------------------------------
        # PHASE 2: histogram-standardize + resize + radiomics, using the cached N4+crop output.
        # ------------------------------------------------------------------

        def finish_clean_case(preprocess, n4_cropped_case):
            """Everything AFTER N4+crop — histogram standardization + resize."""
            case = preprocess.transform_histogram_standardization(n4_cropped_case)
            case = preprocess.resize_planes(case, target_size=(224, 224))
            return case

        print("Building cache ...")
        for i, case in enumerate(dataset):
            cache_path = CACHE_ROOT / f"{case['case_id']}.npz"
            if cache_path.exists():
                continue

            print(f"[Final {i + 1}/{len(dataset)}] Processing case {case['case_id']}...")

            n4_cropped = load_n4_cropped(case)
            cleaned = finish_clean_case(preprocess, n4_cropped)
            radiomics_feats = radiomics.extract(cleaned)

            # Save the fully cleaned case in cache
            atomic_save(
                cache_path,
                case_id=cleaned["case_id"],
                axial=cleaned["planes"]["axial"],
                coronal=cleaned["planes"]["coronal"],
                sagittal=cleaned["planes"]["sagittal"],
                radiomics_axial=radiomics_feats["axial"],
                radiomics_coronal=radiomics_feats["coronal"],
                radiomics_sagittal=radiomics_feats["sagittal"],
                label_abnormal=case["labels"]["abnormal"],
                label_acl=case["labels"]["acl"],
                label_meniscus=case["labels"]["meniscus"],
                split=case["split"],
                cv_fold=case["cv_fold"],
            )

        print("Done. You can delete ./Cache_n4cropped/ now if you want to free disk space.")
        print("It was only needed as an intermediate step and isn't used during training.")



    # ------------------------------------------------------------------
    # PHASE 3: train all 9 plane/label CNNs
    # ------------------------------------------------------------------
    from Model import train as train_plane_model

    PLANES = ["axial", "coronal", "sagittal"]
    LABELS = ["abnormal", "acl", "meniscus"]
    EPOCHS = 20
    LR = 1e-5

    for label in LABELS:
        for plane in PLANES:
            print(f"\n{'='*60}\nTraining {plane} / {label}\n{'='*60}")
            train_plane_model(plane, label, EPOCHS, LR)


    # ------------------------------------------------------------------
    # PHASE 4: combine the 3 planes per label by simple averaging.
    # ------------------------------------------------------------------
    from Metrics import Metrics

    def metrics_from_columns(y_true, y_scores, threshold=0.5):
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        return Metrics(tp, fp, tn, fn, y_true=y_true, y_scores=y_scores)

    print(f"\n{'='*60}\nEnsemble results (simple average across planes)\n{'='*60}")

    for label in LABELS:
        base = pd.read_csv(f"predictions_axial_{label}.csv")
        base = base[["case_id", "true_label", "prob"]].rename(columns={"prob": "prob_axial"})

        for plane in ["coronal", "sagittal"]:
            df = pd.read_csv(f"predictions_{plane}_{label}.csv")
            df = df[["case_id", "prob"]].rename(columns={"prob": f"prob_{plane}"})
            base = base.merge(df, on="case_id")

        base["prob_avg"] = base[["prob_axial", "prob_coronal", "prob_sagittal"]].mean(axis=1)

        print(f"\n[{label}] ENSEMBLE (average across planes) — test set")
        print(metrics_from_columns(base["true_label"], base["prob_avg"]))

        for plane in PLANES:
            print(f"\n[{label}] {plane} alone — test set")
            print(metrics_from_columns(base["true_label"], base[f"prob_{plane}"]))