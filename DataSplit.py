"""

Splits a list of case dicts into train/val/test, stratified jointly on all three labels, plus stratified
CV folds within train.
Returns the SAME dicts with a "split" key added.


Expected input, a list of:
    {
        "case_id": "0000",
        "paths": {"axial": Path(...), "coronal": Path(...), "sagittal": Path(...)},
        "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
    }

"""


import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold


LABEL_KEYS = ["meniscus", "acl", "abnormal"]
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
NUM_CV_FOLDS = 5

class DataSplit:

    def __init__(self, dataset):
        self.dataset = dataset


    def split_dataset(self):

        """
            Returns a NEW list of the same dicts, each with an added
            "split" key ("train" / "val" / "test") and "cv_fold" key
        """

        case_ids = [ case["case_id"] for case in self.dataset ]
        y = np.array( [[case["labels"][label] for label in LABEL_KEYS] for case in self.dataset] )
        idx = np.arange(len(self.dataset))

        # Carve test set stratified on all 3 labels jointly
        msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        trainval_idx, test_idx = next(msss_test.split(idx,y))

        # Split remaining pool into train/val
        y_trainval = y[trainval_idx]
        val_frac_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
        msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=RANDOM_STATE)
        train_sub_idx, val_sub_idx = next(msss_val.split(trainval_idx, y_trainval))
        train_idx = trainval_idx[train_sub_idx]
        val_idx = trainval_idx[val_sub_idx]

        split_of = {}
        for i in train_idx: split_of[case_ids[i]] = "train"
        for i in val_idx:   split_of[case_ids[i]] = "val"
        for i in test_idx:  split_of[case_ids[i]] = "test"

        # Stratified CV folds within train only
        fold_of = {cid: -1 for cid in case_ids}
        y_train = y[train_idx]
        mskf = MultilabelStratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (_, val_fold_sub_idx) in enumerate(mskf.split(train_idx, y_train)):
            for j in val_fold_sub_idx:
                fold_of[case_ids[train_idx[j]]] = fold

        """
        cv_fold: which of the 5 CV rounds this case is held out for validation
        (-1 if not applicable, i.e. case is in val/test, not train).
        """
        out = []
        for c in self.dataset:
            c = dict(c)
            c["split"] = split_of[c["case_id"]]
            c["cv_fold"] = fold_of[c["case_id"]]
            out.append(c)
        return out

    @staticmethod
    def report(cases):
        df = pd.DataFrame([
            {"case_id": c["case_id"], "split": c["split"], **c["labels"]}
            for c in cases
        ])
        print(f"Total cases: {len(df)}\n")
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            n = len(sub)
            print(f"{split:5s} n={n:4d} ({n / len(df):.1%})")
            for label in LABEL_KEYS:
                pos = sub[label].sum()
                print(f"    {label:9s}: {pos:4d} pos / {n - pos:4d} neg  ({pos / n:.1%} pos)")
        print("\nFull dataset for comparison:")
        for label in LABEL_KEYS:
            pos = df[label].sum()
            print(f"    {label:9s}: {pos:4d} pos / {len(df) - pos:4d} neg  ({pos / len(df):.1%} pos)")

    @staticmethod
    def compute_pos_weights(cases, split="train", exclude_fold=None):
        sub = [c for c in cases if c["split"] == split]
        if exclude_fold is not None:
            sub = [c for c in sub if c["cv_fold"] != exclude_fold]
        weights = {}
        for label in LABEL_KEYS:
            pos = sum(c["labels"][label] for c in sub)
            neg = len(sub) - pos
            weights[label] = neg / pos
        return weights