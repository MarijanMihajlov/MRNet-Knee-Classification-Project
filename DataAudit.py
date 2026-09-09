from collections import defaultdict

import numpy as np
from pathlib import Path
from typing import Dict, Any, List


class DataAudit:
    """
        Data auditor for MRNet-style datasets.

        Expected dataset format:
        [
            {
                "case_id": "0000",
                "paths": {
                    "axial": Path(".../train/axial/0000.npy"),
                    "coronal": Path(".../train/coronal/0000.npy"),
                    "sagittal": Path(".../train/sagittal/0000.npy"),
                },
                "labels": {"abnormal": 1, "acl": 0, "meniscus": 1},
            },
            ...
        ]
    """

    EXPECTED_PLANES = {"axial", "coronal", "sagittal"}
    EXPECTED_LABELS = {"abnormal", "acl", "meniscus"}

    def __init__(self, dataset):
        self.dataset = dataset

        # Accumulators
        self._issues = defaultdict(list)
        self._seen_ids = set()
        self._raw_stats = {
            "slice_counts": defaultdict(list),
            "spatial_shapes": defaultdict(list),
            "intensity_mins": [],
            "intensity_maxs": [],
            "intensity_means": [],
            "label_counts": {k: [0, 0] for k in self.EXPECTED_LABELS},
        }

    # ------------------------------------------------------------------ #
    # Per-item check functions
    # ------------------------------------------------------------------ #

    def _safe_load(self, path: Path):
        """Load a .npy file and return (array, error_message)."""
        try:
            arr = np.load(path, allow_pickle=False)
            return arr, None
        except Exception as e:
            return None, str(e)

    def check_file_existence(self, case: Dict[str, Any]) -> List[Dict]:
        """Verify every path in the case points to an existing file."""
        issues = []
        cid = case["case_id"]
        for plane, path in case["paths"].items():
            if not path.exists():
                issues.append({
                    "case_id": cid, "plane": plane,
                    "path": str(path), "check": "file_existence"
                })
        return issues

    def check_path_validity(self, case: Dict[str, Any]) -> List[Dict]:
        """Ensure the filename matches the case_id."""
        issues = []
        cid = case["case_id"]
        expected = f"{cid}.npy"
        for plane, path in case["paths"].items():
            if path.name != expected:
                issues.append({
                    "case_id": cid, "plane": plane,
                    "expected": expected, "actual": path.name,
                    "check": "path_validity"
                })
        return issues

    def check_plane_completeness(self, case: Dict[str, Any]) -> List[Dict]:
        """Ensure the case has exactly the three expected planes."""
        issues = []
        actual = set(case["paths"].keys())
        if actual != self.EXPECTED_PLANES:
            issues.append({
                "case_id": case["case_id"],
                "missing": list(self.EXPECTED_PLANES - actual),
                "extra": list(actual - self.EXPECTED_PLANES),
                "check": "plane_completeness"
            })
        return issues

    def check_labels(self, case: Dict[str, Any]) -> List[Dict]:
        """
        Validate labels for a single case:
        - Correct keys present
        - Binary (0 or 1) values
        - Logical constraint: abnormal=0 => acl=0 AND meniscus=0
        Also updates internal label statistics.
        """
        issues = []
        cid = case["case_id"]
        labels = case["labels"]

        # Duplicate ID?
        if cid in self._seen_ids:
            issues.append({"case_id": cid, "check": "duplicate_case_id"})
        self._seen_ids.add(cid)

        # Keys match?
        actual_keys = set(labels.keys())
        if actual_keys != self.EXPECTED_LABELS:
            issues.append({
                "case_id": cid,
                "missing": list(self.EXPECTED_LABELS - actual_keys),
                "extra": list(actual_keys - self.EXPECTED_LABELS),
                "check": "label_keys"
            })
            return issues  # Cannot validate further if keys are wrong

        # Values & stats
        for name in self.EXPECTED_LABELS:
            val = labels[name]
            if val not in (0, 1):
                issues.append({
                    "case_id": cid, "label": name,
                    "value": val, "check": "label_binary"
                })
            else:
                self._raw_stats["label_counts"][name][val] += 1

        # Logic constraint
        if labels.get("abnormal") == 0:
            if labels.get("acl") == 1 or labels.get("meniscus") == 1:
                issues.append({
                    "case_id": cid,
                    "labels": {k: labels[k] for k in self.EXPECTED_LABELS},
                    "check": "label_logic"
                })

        return issues

    def check_array_integrity(self, case: Dict[str, Any]) -> List[Dict]:
        """
        Load every .npy file for this case and validate shape, dtype,
        emptiness, and invalid values (NaN/Inf).
        Also updates internal intensity/shape statistics.
        """
        issues = []
        cid = case["case_id"]

        for plane, path in case["paths"].items():
            if not path.exists():
                continue  # Handled by check_file_existence

            try:
                arr = np.load(path, allow_pickle=False)
            except Exception as e:
                issues.append({
                    "case_id": cid, "plane": plane,
                    "error": str(e), "check": "array_load"
                })
                continue

            # Dimensionality
            if arr.ndim != 3:
                issues.append({
                    "case_id": cid, "plane": plane,
                    "shape": arr.shape, "ndim": arr.ndim,
                    "check": "array_dimensions"
                })
                continue

            # Dtype
            if not np.issubdtype(arr.dtype, np.number):
                issues.append({
                    "case_id": cid, "plane": plane,
                    "dtype": str(arr.dtype), "check": "array_dtype"
                })
                continue

            # Empty
            if arr.size == 0:
                issues.append({
                    "case_id": cid, "plane": plane,
                    "shape": arr.shape, "check": "array_empty"
                })
                continue

            # NaN / Inf
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            if has_nan or has_inf:
                issues.append({
                    "case_id": cid, "plane": plane,
                    "has_nan": bool(has_nan), "has_inf": bool(has_inf),
                    "nan_pixels": int(np.isnan(arr).sum()) if has_nan else 0,
                    "inf_pixels": int(np.isinf(arr).sum()) if has_inf else 0,
                    "check": "array_invalid_values"
                })

            # ---- Statistics (side effects) ----
            s, h, w = arr.shape
            self._raw_stats["slice_counts"][plane].append(s)
            self._raw_stats["spatial_shapes"][plane].append((h, w))
            self._raw_stats["intensity_mins"].append(float(arr.min()))
            self._raw_stats["intensity_maxs"].append(float(arr.max()))
            self._raw_stats["intensity_means"].append(float(arr.mean()))

        return issues

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def execute(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Single-pass execution: iterate dataset once, run all checks per case.
        Returns the full audit report.
        """
        if verbose:
            print("Starting MRNet Data Audit (single-pass)...")

        # Register all check functions here
        check_functions = [
            self.check_file_existence,
            self.check_path_validity,
            self.check_plane_completeness,
            self.check_labels,
            self.check_array_integrity,
        ]

        for case in self.dataset:
            for fn in check_functions:
                found_issues = fn(case)
                for issue in found_issues:
                    self._issues[issue["check"]].append(issue)

        report = self._build_report()
        if verbose:
            self._print_report(report)
        return report

    # ------------------------------------------------------------------ #
    # Report builders
    # ------------------------------------------------------------------ #

    def _build_report(self) -> Dict[str, Any]:
        total_issues = sum(len(v) for v in self._issues.values())

        report = {
            "summary": {
                "total_cases": len(self.dataset),
                "total_issues": total_issues,
                "status": "PASS" if total_issues == 0 else "FAIL",
                "issue_categories": list(self._issues.keys()),
            },
            "issues": dict(self._issues),
            "statistics": self._compile_statistics(),
        }
        return report

    def _compile_statistics(self) -> Dict[str, Any]:
        stats = {}

        # Slice counts
        if self._raw_stats["slice_counts"]:
            stats["slice_counts"] = {
                plane: {
                    "min": int(np.min(vals)),
                    "max": int(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                }
                for plane, vals in self._raw_stats["slice_counts"].items()
            }

        # Spatial shapes
        if self._raw_stats["spatial_shapes"]:
            stats["spatial_shapes"] = {
                plane: {
                    "unique_shapes": list(set(vals)),
                    "count": len(vals),
                }
                for plane, vals in self._raw_stats["spatial_shapes"].items()
            }

        # Intensity
        if self._raw_stats["intensity_mins"]:
            stats["intensity"] = {
                "global_min": float(np.min(self._raw_stats["intensity_mins"])),
                "global_max": float(np.max(self._raw_stats["intensity_maxs"])),
                "mean_of_means": float(np.mean(self._raw_stats["intensity_means"])),
            }

        # Labels
        stats["label_distribution"] = {
            name: {"negative": c[0], "positive": c[1]}
            for name, c in self._raw_stats["label_counts"].items()
        }

        return stats

    def _print_report(self, report: Dict[str, Any]):
        s = report["summary"]
        print("\n" + "=" * 60)
        print(f" AUDIT RESULT: {s['status']}")
        print("=" * 60)
        print(f"Cases audited : {s['total_cases']}")
        print(f"Issues found  : {s['total_issues']}")

        if s["issue_categories"]:
            print("\nBreakdown by check:")
            for cat in s["issue_categories"]:
                count = len(report["issues"][cat])
                print(f"  • {cat:25s}: {count:4d}")
                # Show first example
                if report["issues"][cat]:
                    print(f"      e.g. {report['issues'][cat][0]}")

        st = report["statistics"]
        if "slice_counts" in st:
            print("\nSlice counts per plane:")
            for plane, vals in st["slice_counts"].items():
                print(f"  {plane:10s}: min={vals['min']:3d}  max={vals['max']:3d}  "
                      f"mean={vals['mean']:.1f}  median={vals['median']:.1f}")

        if "label_distribution" in st:
            print("\nLabel distribution:")
            for lbl, d in st["label_distribution"].items():
                total = d["negative"] + d["positive"]
                pos_pct = 100 * d["positive"] / total if total else 0
                print(f"  {lbl:10s}: {d['negative']:4d} neg / {d['positive']:4d} pos  "
                      f"({pos_pct:.1f}%)")

        if "intensity" in st:
            i = st["intensity"]
            print(f"\nIntensity: [{i['global_min']:.2f}, {i['global_max']:.2f}]  "
                  f"mean={i['mean_of_means']:.2f}")

        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Post-audit helpers
    # ------------------------------------------------------------------ #

    def get_failed_case_ids(self, report: Dict[str, Any]) -> set:
        """Extract case IDs that triggered at least one issue."""
        failed = set()
        for issues in report["issues"].values():
            for issue in issues:
                if "case_id" in issue:
                    failed.add(issue["case_id"])
        return failed

    def get_clean_cases(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return cases that passed every check."""
        failed = self.get_failed_case_ids(report)
        return [c for c in self.dataset if c["case_id"] not in failed]




if __name__ == "__main__":
    auditor = DataAudit(dataset)
    report = auditor.execute(verbose=True)

    # Filter downstream
    clean = auditor.get_clean_cases(report)
    print(f"Clean cases: {len(clean)} / {len(dataset)}")

    # Inspect failures
    if report["issues"].get("label_logic"):
        print("Label leakage:", report["issues"]["label_logic"])