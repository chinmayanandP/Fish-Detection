from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_ROOT = PROJECT_ROOT / "preprocessing"

PREPARED_ROOT = PREPROCESSING_ROOT / "prepared_dataset"
PREPROCESSED_ROOT = PREPROCESSING_ROOT / "preprocessed_dataset"

MASTER_SMOKE_ROOT = (
    PREPROCESSING_ROOT
    / "master_preprocessing_smoke"
)

IMAGE_INDEX_FILE = (
    PREPARED_ROOT
    / "metadata"
    / "image_index.json"
)

DATASET_INFO_FILE = (
    PREPARED_ROOT
    / "metadata"
    / "dataset_info.json"
)

PREPARED_DATASET_YAML = (
    PREPARED_ROOT
    / "dataset.yaml"
)

CERT_REPORTS_ROOT = (
    PREPROCESSING_ROOT
    / "smoke_test_results"
    / "reports"
)

SMOKE_REPORT_FILE = (
    CERT_REPORTS_ROOT
    / "smoke_test_report.json"
)

CALIBRATION_REPORT_FILE = (
    CERT_REPORTS_ROOT
    / "calibration_report.json"
)

SAMPLING_REPORT_FILE = (
    CERT_REPORTS_ROOT
    / "sampling_independence_report.json"
)

SMOKE_TEST_SOURCE_FILE = (
    PREPROCESSING_ROOT
    / "smoke_test.py"
)


# ============================================================
# FROZEN PREPROCESSING POLICY
# ============================================================

FROZEN_POLICY_VERSION = "1.4.3"

FROZEN_SELECTOR_SOURCE_VERSION = (
    "1.4.3-calibration"
)

FROZEN_MINIMUM_IMPROVEMENT = 0.0200

FROZEN_DEPLOYMENT_CAP = 6

FROZEN_SMOKE_IMAGE_COUNT = 20
FROZEN_CALIBRATION_IMAGE_COUNT = 200

FROZEN_SMOKE_DEPLOYMENT_COUNT = 10
FROZEN_CALIBRATION_DEPLOYMENT_COUNT = 51

FROZEN_VISUAL_REVIEW_STATUS = "APPROVED"


# ============================================================
# OUTPUT POLICY
# ============================================================

# Raw winner:
#     copy the prepared source image byte-for-byte.
#
# Processed winner:
#     write losslessly as PNG.
#
# Labels:
#     copy byte-for-byte.
#
# No resizing.
# No cropping.
# No geometric modification.

PROCESSED_IMAGE_EXTENSION = ".png"

PNG_COMPRESSION = 3

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# IMPORT THE CERTIFIED SELECTOR
#
# IMPORTANT:
# We intentionally reuse smoke_test.py.
#
# We do NOT duplicate:
# - preprocessing candidates
# - safety gates
# - local detail protection
# - benefit scoring
# - winner selection
#
# This prevents implementation drift between calibration and
# production preprocessing.
# ============================================================

if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PREPROCESSING_ROOT),
    )

from smoke_test import (  # type: ignore  # noqa: E402
    PREPROCESSING_MODES,
    select_best_candidate,
)
from smoke_test import (
    PREPROCESSING_POLICY_VERSION as SELECTOR_SOURCE_VERSION,
)

# ============================================================
# GENERAL UTILITIES
# ============================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def print_header(
    title: str,
) -> None:
    print()

    print(
        "=" * 82
    )

    print(
        f" {title}"
    )

    print(
        "=" * 82
    )


def read_json(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_json(
    path: Path,
    data: object,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_hash(
    data: object,
) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name
        + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    shutil.copyfile(
        source,
        temporary,
    )

    os.replace(
        temporary,
        destination,
    )


def atomic_write_png(
    image: np.ndarray,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.stem
        + ".tmp"
        + destination.suffix
    )

    if temporary.exists():
        temporary.unlink()

    success = cv2.imwrite(
        str(
            temporary
        ),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            PNG_COMPRESSION,
        ],
    )

    if not success:
        raise RuntimeError(
            "Failed to write PNG: "
            f"{temporary}"
        )

    os.replace(
        temporary,
        destination,
    )


def relative_path(
    path: Path,
    root: Path,
) -> str:
    return path.relative_to(
        root
    ).as_posix()


def number_close(
    value,
    expected: float,
) -> bool:
    try:
        return math.isclose(
            float(
                value
            ),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def report_passed(
    report: dict,
) -> bool:
    status = str(
        report.get(
            "status",
            ""
        )
    ).strip().upper()

    if status in {
        "PASS",
        "PASSED",
        "CERTIFIED",
    }:
        return True

    passed = report.get(
        "passed"
    )

    return passed is True


# ============================================================
# SOURCE METADATA
# ============================================================


def load_source_records() -> tuple[
    list[dict],
    dict,
]:
    image_index = read_json(
        IMAGE_INDEX_FILE
    )

    if not isinstance(
        image_index,
        list,
    ):
        raise RuntimeError(
            "image_index.json must contain a list."
        )

    if DATASET_INFO_FILE.exists():
        dataset_info = read_json(
            DATASET_INFO_FILE
        )
    else:
        dataset_info = {}

    if not isinstance(
        dataset_info,
        dict,
    ):
        dataset_info = {}

    return (
        image_index,
        dataset_info,
    )


# ============================================================
# FROZEN POLICY METADATA
# ============================================================


def build_frozen_policy(
    dataset_info: dict,
) -> dict:
    policy = {
        "policy_version":
            FROZEN_POLICY_VERSION,

        "selector_source_version":
            FROZEN_SELECTOR_SOURCE_VERSION,

        "selector_file_sha256":
            sha256_file(
                SMOKE_TEST_SOURCE_FILE
            ),

        "minimum_improvement":
            FROZEN_MINIMUM_IMPROVEMENT,

        "candidate_modes":
            list(
                PREPROCESSING_MODES
            ),

        "runtime_selector":
            "label_free",

        "annotations_used_at_runtime":
            False,

        "raw_fallback":
            True,

        "raw_encoding":
            "byte_for_byte_source_copy",

        "processed_encoding":
            "lossless_png",

        "label_handling":
            "byte_for_byte_copy",

        "visual_review_status":
            FROZEN_VISUAL_REVIEW_STATUS,

        "source_dataset_fingerprint_sha256":
            dataset_info.get(
                "dataset_fingerprint_sha256"
            ),
    }

    policy[
        "policy_fingerprint_sha256"
    ] = canonical_hash(
        policy
    )

    return policy


# ============================================================
# MASTER PREFLIGHT
# ============================================================


def preflight(
    image_index: list[dict],
    dataset_info: dict,
) -> dict:
    print_header(
        "MASTER PREPROCESSING PREFLIGHT"
    )

    # --------------------------------------------------------
    # 1. Certified selector version
    # --------------------------------------------------------

    if (
        SELECTOR_SOURCE_VERSION
        != FROZEN_SELECTOR_SOURCE_VERSION
    ):
        raise RuntimeError(
            "Selector version drift detected. "
            f"Expected {FROZEN_SELECTOR_SOURCE_VERSION}; "
            f"found {SELECTOR_SOURCE_VERSION}."
        )

    # --------------------------------------------------------
    # 2. Candidate-mode count
    # --------------------------------------------------------

    if len(
        PREPROCESSING_MODES
    ) != 8:
        raise RuntimeError(
            "Expected exactly 8 certified preprocessing "
            f"modes, found {len(PREPROCESSING_MODES)}."
        )

    # --------------------------------------------------------
    # 3. Load certification reports
    # --------------------------------------------------------

    smoke_report = read_json(
        SMOKE_REPORT_FILE
    )

    calibration_report = read_json(
        CALIBRATION_REPORT_FILE
    )

    sampling_report = read_json(
        SAMPLING_REPORT_FILE
    )

    for (
        name,
        report,
    ) in (
        (
            "smoke_test_report.json",
            smoke_report,
        ),
        (
            "calibration_report.json",
            calibration_report,
        ),
        (
            "sampling_independence_report.json",
            sampling_report,
        ),
    ):
        if not isinstance(
            report,
            dict,
        ):
            raise RuntimeError(
                f"{name} must contain a JSON object."
            )

    # --------------------------------------------------------
    # 4. Certification versions
    # --------------------------------------------------------

    if (
        smoke_report.get(
            "policy_version"
        )
        != FROZEN_SELECTOR_SOURCE_VERSION
    ):
        raise RuntimeError(
            "Smoke report policy version mismatch."
        )

    if (
        calibration_report.get(
            "policy_version"
        )
        != FROZEN_SELECTOR_SOURCE_VERSION
    ):
        raise RuntimeError(
            "Calibration report policy version mismatch."
        )

    if (
        sampling_report.get(
            "policy_version"
        )
        != FROZEN_SELECTOR_SOURCE_VERSION
    ):
        raise RuntimeError(
            "Sampling-independence report policy "
            "version mismatch."
        )

    # --------------------------------------------------------
    # 5. Frozen threshold
    # --------------------------------------------------------

    if not number_close(
        smoke_report.get(
            "calibrated_minimum_improvement"
        ),
        FROZEN_MINIMUM_IMPROVEMENT,
    ):
        raise RuntimeError(
            "Smoke certification threshold "
            "does not equal frozen 0.0200."
        )

    if not number_close(
        calibration_report.get(
            "selected_threshold"
        ),
        FROZEN_MINIMUM_IMPROVEMENT,
    ):
        raise RuntimeError(
            "Calibration threshold does not "
            "equal frozen 0.0200."
        )

    # --------------------------------------------------------
    # 6. Runtime selector must remain label-free
    # --------------------------------------------------------

    if (
        smoke_report.get(
            "runtime_selector_label_free"
        )
        is not True
    ):
        raise RuntimeError(
            "Smoke report does not certify "
            "a label-free runtime selector."
        )

    if (
        calibration_report.get(
            "runtime_selector"
        )
        != "label_free"
    ):
        raise RuntimeError(
            "Calibration report does not certify "
            "a label-free runtime selector."
        )

    if (
        calibration_report.get(
            "annotations_used_for_runtime_selection"
        )
        is not False
    ):
        raise RuntimeError(
            "Calibration report indicates that annotations "
            "may have been used at runtime."
        )

    # --------------------------------------------------------
    # 7. Class-neutral threshold calibration
    # --------------------------------------------------------

    if (
        calibration_report.get(
            "class_labels_used_for_threshold_reward"
        )
        is not False
    ):
        raise RuntimeError(
            "Calibration threshold was not class-neutral."
        )

    if (
        calibration_report.get(
            "positive_images_rewarded"
        )
        is not False
    ):
        raise RuntimeError(
            "Positive images were rewarded during "
            "threshold calibration."
        )

    if (
        calibration_report.get(
            "negative_images_penalized"
        )
        is not False
    ):
        raise RuntimeError(
            "Negative images were penalized during "
            "threshold calibration."
        )

    # --------------------------------------------------------
    # 8. Calibration safe
    # --------------------------------------------------------

    if (
        calibration_report.get(
            "calibration_safe"
        )
        is not True
    ):
        raise RuntimeError(
            "Calibration report is not marked safe."
        )

    threshold_analysis = (
        calibration_report.get(
            "threshold_analysis",
            []
        )
    )

    if not isinstance(
        threshold_analysis,
        list,
    ):
        raise RuntimeError(
            "Calibration threshold_analysis "
            "must be a list."
        )

    selected_rows = [
        row
        for row
        in threshold_analysis
        if (
            isinstance(
                row,
                dict,
            )
            and number_close(
                row.get(
                    "threshold"
                ),
                FROZEN_MINIMUM_IMPROVEMENT,
            )
        )
    ]

    if len(
        selected_rows
    ) != 1:
        raise RuntimeError(
            "Expected exactly one threshold-analysis "
            "row for frozen threshold 0.0200."
        )

    selected_threshold_row = (
        selected_rows[
            0
        ]
    )

    if int(
        selected_threshold_row.get(
            "roi_unsafe_selected",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Frozen threshold has non-zero "
            "selected ROI failures."
        )

    if (
        selected_threshold_row.get(
            "eligible_for_selection"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen threshold 0.0200 was not "
            "eligible for selection."
        )

    # --------------------------------------------------------
    # 9. Independent smoke winner ROI failures
    # --------------------------------------------------------

    if int(
        smoke_report.get(
            "winner_roi_audit_failures",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Independent smoke test contains "
            "selected winner ROI failures."
        )

    # --------------------------------------------------------
    # 10. Sampling/deployment independence
    #
    # Actual v1.4.3 report structure:
    #
    # {
    #   "status": "PASS",
    #   "policy": {...},
    #   "counts": {...},
    #   "checks": {...}
    # }
    # --------------------------------------------------------

    if not report_passed(
        sampling_report
    ):
        raise RuntimeError(
            "Sampling-independence certification "
            "did not pass."
        )

    sampling_policy = (
        sampling_report.get(
            "policy",
            {}
        )
    )

    sampling_counts = (
        sampling_report.get(
            "counts",
            {}
        )
    )

    sampling_checks = (
        sampling_report.get(
            "checks",
            {}
        )
    )

    if not isinstance(
        sampling_policy,
        dict,
    ):
        raise RuntimeError(
            "sampling report 'policy' "
            "section is invalid."
        )

    if not isinstance(
        sampling_counts,
        dict,
    ):
        raise RuntimeError(
            "sampling report 'counts' "
            "section is invalid."
        )

    if not isinstance(
        sampling_checks,
        dict,
    ):
        raise RuntimeError(
            "sampling report 'checks' "
            "section is invalid."
        )

    # --------------------------------------------------------
    # 10A. Smoke deployments excluded
    # --------------------------------------------------------

    if (
        sampling_policy.get(
            "smoke_deployments_excluded_from_calibration"
        )
        is not True
    ):
        raise RuntimeError(
            "Smoke deployments were not certified as "
            "excluded from calibration."
        )

    if (
        sampling_policy.get(
            "group_aware_round_robin"
        )
        is not True
    ):
        raise RuntimeError(
            "Group-aware calibration sampling "
            "was not certified."
        )

    if (
        sampling_policy.get(
            "positive_and_negative_share_deployment_cap"
        )
        is not True
    ):
        raise RuntimeError(
            "Positive/negative calibration sampling did "
            "not share the deployment cap."
        )

    # --------------------------------------------------------
    # 10B. Exact-image independence
    # --------------------------------------------------------

    image_overlap_count = (
        sampling_checks.get(
            "image_overlap_count"
        )
    )

    if image_overlap_count is None:
        raise RuntimeError(
            "Certified image-overlap count is missing."
        )

    if int(
        image_overlap_count
    ) != 0:
        raise RuntimeError(
            "Certified exact-image overlap is not zero."
        )

    if (
        sampling_checks.get(
            "image_overlap_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Exact-image independence check "
            "did not pass."
        )

    image_overlap_list = (
        sampling_checks.get(
            "image_overlap",
            []
        )
    )

    if not isinstance(
        image_overlap_list,
        list,
    ):
        raise RuntimeError(
            "image_overlap must be a list."
        )

    if image_overlap_list:
        raise RuntimeError(
            "Certified image-overlap list "
            "is not empty."
        )

    # --------------------------------------------------------
    # 10C. Deployment independence
    # --------------------------------------------------------

    deployment_overlap_count = (
        sampling_checks.get(
            "deployment_overlap_count"
        )
    )

    if deployment_overlap_count is None:
        raise RuntimeError(
            "Certified deployment-overlap "
            "count is missing."
        )

    if int(
        deployment_overlap_count
    ) != 0:
        raise RuntimeError(
            "Certified deployment overlap "
            "is not zero."
        )

    if (
        sampling_checks.get(
            "deployment_overlap_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Deployment-independence check "
            "did not pass."
        )

    deployment_overlap_list = (
        sampling_checks.get(
            "deployment_overlap",
            []
        )
    )

    if not isinstance(
        deployment_overlap_list,
        list,
    ):
        raise RuntimeError(
            "deployment_overlap must be a list."
        )

    if deployment_overlap_list:
        raise RuntimeError(
            "Certified deployment-overlap "
            "list is not empty."
        )

    # --------------------------------------------------------
    # 10D. Deployment cap
    # --------------------------------------------------------

    configured_deployment_cap = (
        sampling_policy.get(
            "maximum_calibration_images_per_deployment"
        )
    )

    if configured_deployment_cap is None:
        raise RuntimeError(
            "Certified calibration deployment "
            "cap is missing."
        )

    if int(
        configured_deployment_cap
    ) != FROZEN_DEPLOYMENT_CAP:
        raise RuntimeError(
            "Certified calibration deployment cap "
            "does not equal frozen cap 6."
        )

    observed_max_per_deployment = (
        sampling_counts.get(
            "maximum_observed_calibration_images_per_deployment"
        )
    )

    if observed_max_per_deployment is None:
        raise RuntimeError(
            "Observed maximum calibration images "
            "per deployment is missing."
        )

    if int(
        observed_max_per_deployment
    ) > FROZEN_DEPLOYMENT_CAP:
        raise RuntimeError(
            "Observed calibration deployment usage "
            "exceeds frozen cap 6."
        )

    if (
        sampling_checks.get(
            "deployment_cap_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Deployment-cap certification "
            "did not pass."
        )

    deployment_cap_violations = (
        sampling_checks.get(
            "deployment_cap_violations",
            {},
        )
    )

    if not isinstance(
        deployment_cap_violations,
        dict,
    ):
        raise RuntimeError(
            "deployment_cap_violations "
            "must be an object."
        )

    if deployment_cap_violations:
        raise RuntimeError(
            "Deployment-cap violations are present: "
            f"{deployment_cap_violations}"
        )

    # --------------------------------------------------------
    # 10E. Certified sample counts
    # --------------------------------------------------------

    smoke_images = (
        sampling_counts.get(
            "smoke_images"
        )
    )

    calibration_images = (
        sampling_counts.get(
            "calibration_images"
        )
    )

    smoke_deployments = (
        sampling_counts.get(
            "smoke_deployments"
        )
    )

    calibration_deployments = (
        sampling_counts.get(
            "calibration_deployments"
        )
    )

    if int(
        smoke_images
        if smoke_images is not None
        else -1
    ) != FROZEN_SMOKE_IMAGE_COUNT:
        raise RuntimeError(
            "Certified smoke image count "
            "does not equal 20."
        )

    if int(
        calibration_images
        if calibration_images is not None
        else -1
    ) != FROZEN_CALIBRATION_IMAGE_COUNT:
        raise RuntimeError(
            "Certified calibration image count "
            "does not equal 200."
        )

    if int(
        smoke_deployments
        if smoke_deployments is not None
        else -1
    ) != FROZEN_SMOKE_DEPLOYMENT_COUNT:
        raise RuntimeError(
            "Certified smoke deployment count "
            "does not equal 10."
        )

    if int(
        calibration_deployments
        if calibration_deployments is not None
        else -1
    ) != FROZEN_CALIBRATION_DEPLOYMENT_COUNT:
        raise RuntimeError(
            "Certified calibration deployment "
            "count does not equal 51."
        )

    # --------------------------------------------------------
    # 10F. Verify deployment-count map itself
    # --------------------------------------------------------

    deployment_counts = (
        sampling_report.get(
            "calibration_deployment_counts",
            {}
        )
    )

    if not isinstance(
        deployment_counts,
        dict,
    ):
        raise RuntimeError(
            "calibration_deployment_counts "
            "must be an object."
        )

    if len(
        deployment_counts
    ) != FROZEN_CALIBRATION_DEPLOYMENT_COUNT:
        raise RuntimeError(
            "calibration_deployment_counts does not "
            "contain 51 deployments."
        )

    if deployment_counts:
        calculated_max = max(
            int(
                value
            )
            for value
            in deployment_counts.values()
        )
    else:
        calculated_max = 0

    if (
        calculated_max
        != int(
            observed_max_per_deployment
        )
    ):
        raise RuntimeError(
            "Reported maximum deployment usage does "
            "not match calibration_deployment_counts."
        )

    if (
        calculated_max
        > FROZEN_DEPLOYMENT_CAP
    ):
        raise RuntimeError(
            "Calibration deployment-count map "
            "violates cap 6."
        )

    # --------------------------------------------------------
    # 11. Prepared dataset split integrity
    # --------------------------------------------------------

    required_splits = {
        "train",
        "val",
        "test",
    }

    found_splits = {
        str(
            record.get(
                "split"
            )
        )
        for record
        in image_index
    }

    if not required_splits.issubset(
        found_splits
    ):
        raise RuntimeError(
            "Prepared dataset splits are incomplete: "
            f"{sorted(found_splits)}"
        )

    # --------------------------------------------------------
    # 12. Prepared dataset file integrity
    # --------------------------------------------------------

    seen_images = set()
    seen_labels = set()

    split_counts = Counter()

    for record in image_index:
        image_rel = str(
            record.get(
                "prepared_image",
                "",
            )
        )

        label_rel = str(
            record.get(
                "prepared_label",
                "",
            )
        )

        split = str(
            record.get(
                "split",
                "",
            )
        )

        if (
            not image_rel
            or not label_rel
        ):
            raise RuntimeError(
                "Prepared dataset record contains "
                "missing image/label paths."
            )

        if image_rel in seen_images:
            raise RuntimeError(
                "Duplicate prepared image record: "
                f"{image_rel}"
            )

        if label_rel in seen_labels:
            raise RuntimeError(
                "Duplicate prepared label record: "
                f"{label_rel}"
            )

        seen_images.add(
            image_rel
        )

        seen_labels.add(
            label_rel
        )

        source_image = (
            PREPARED_ROOT
            / image_rel
        )

        source_label = (
            PREPARED_ROOT
            / label_rel
        )

        if not source_image.is_file():
            raise FileNotFoundError(
                "Missing prepared image: "
                f"{image_rel}"
            )

        if not source_label.is_file():
            raise FileNotFoundError(
                "Missing prepared label: "
                f"{label_rel}"
            )

        if not image_rel.startswith(
            f"{split}/images/"
        ):
            raise RuntimeError(
                "Prepared image split/path mismatch: "
                f"{image_rel}"
            )

        if not label_rel.startswith(
            f"{split}/labels/"
        ):
            raise RuntimeError(
                "Prepared label split/path mismatch: "
                f"{label_rel}"
            )

        split_counts[
            split
        ] += 1

    # --------------------------------------------------------
    # 13. Expected Stage-1 counts
    # --------------------------------------------------------

    expected_split_counts = {
        "train":
            10181,

        "val":
            2273,

        "test":
            2220,
    }

    for (
        split,
        expected,
    ) in expected_split_counts.items():
        actual = int(
            split_counts[
                split
            ]
        )

        if actual != expected:
            raise RuntimeError(
                f"Prepared {split} count changed. "
                f"Expected {expected}; found {actual}."
            )

    if len(
        image_index
    ) != 14674:
        raise RuntimeError(
            "Prepared dataset image count changed. "
            f"Expected 14,674; found {len(image_index):,}."
        )

    # --------------------------------------------------------
    # 14. Build policy fingerprint
    # --------------------------------------------------------

    policy = build_frozen_policy(
        dataset_info
    )

    # --------------------------------------------------------
    # 15. Human-readable certification
    # --------------------------------------------------------

    print(
        f"Policy source             : "
        f"{SELECTOR_SOURCE_VERSION}"
    )

    print(
        f"Frozen threshold          : "
        f"{FROZEN_MINIMUM_IMPROVEMENT:.4f}"
    )

    print(
        f"Candidate modes           : "
        f"{len(PREPROCESSING_MODES)}"
    )

    print(
        "Runtime annotations       : NONE"
    )

    print(
        "Calibration ROI failures  : 0"
    )

    print(
        "Smoke winner ROI failures : 0"
    )

    print(
        f"Smoke images              : "
        f"{smoke_images}"
    )

    print(
        f"Calibration images        : "
        f"{calibration_images}"
    )

    print(
        f"Smoke deployments         : "
        f"{smoke_deployments}"
    )

    print(
        f"Calibration deployments   : "
        f"{calibration_deployments}"
    )

    print(
        f"Exact image overlap       : "
        f"{image_overlap_count}"
    )

    print(
        f"Deployment overlap        : "
        f"{deployment_overlap_count}"
    )

    print(
        f"Calibration group cap     : "
        f"{configured_deployment_cap}"
    )

    print(
        f"Observed max/deployment   : "
        f"{observed_max_per_deployment}"
    )

    print(
        f"Prepared images           : "
        f"{len(image_index):,}"
    )

    print(
        f"  train                   : "
        f"{split_counts['train']:,}"
    )

    print(
        f"  val                     : "
        f"{split_counts['val']:,}"
    )

    print(
        f"  test                    : "
        f"{split_counts['test']:,}"
    )

    print(
        "Visual review             : APPROVED"
    )

    print(
        "Result                    : PASS"
    )

    return {
        "policy":
            policy,

        "smoke_report":
            smoke_report,

        "calibration_report":
            calibration_report,

        "sampling_report":
            sampling_report,
    }


# ============================================================
# OUTPUT ROOT INITIALIZATION
# ============================================================


def prepare_output_root(
    output_root: Path,
    reset: bool,
) -> None:
    if (
        reset
        and output_root.exists()
    ):
        shutil.rmtree(
            output_root
        )

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            output_root
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_root
            / split
            / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    for directory_name in (
        "manifests",
        "metadata",
        "reports",
    ):
        (
            output_root
            / directory_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATASET YAML
# ============================================================


def write_dataset_yaml(
    output_root: Path,
) -> None:
    names = {
        0:
            "fish"
    }

    if PREPARED_DATASET_YAML.exists():
        source_config = yaml.safe_load(
            PREPARED_DATASET_YAML.read_text(
                encoding="utf-8"
            )
        )

        if (
            isinstance(
                source_config,
                dict,
            )
            and "names" in source_config
        ):
            names = (
                source_config[
                    "names"
                ]
            )

    if isinstance(
        names,
        (
            list,
            dict,
        ),
    ):
        class_count = len(
            names
        )
    else:
        class_count = 1

    config = {
        "path":
            output_root
            .resolve()
            .as_posix(),

        "train":
            "train/images",

        "val":
            "val/images",

        "test":
            "test/images",

        "names":
            names,

        "nc":
            class_count,
    }

    atomic_write_text(
        output_root
        / "dataset.yaml",

        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
    )


# ============================================================
# INITIAL RUN METADATA
# ============================================================


def write_initial_metadata(
    output_root: Path,
    policy: dict,
    dataset_info: dict,
    run_mode: str,
) -> None:
    write_json(
        output_root
        / "metadata"
        / "frozen_policy.json",

        policy,
    )

    write_json(
        output_root
        / "metadata"
        / "source_dataset_info.json",

        dataset_info,
    )

    write_json(
        output_root
        / "metadata"
        / "run_configuration.json",

        {
            "generated_at_utc":
                utc_now(),

            "run_mode":
                run_mode,

            "policy_version":
                FROZEN_POLICY_VERSION,

            "selector_source_version":
                FROZEN_SELECTOR_SOURCE_VERSION,

            "frozen_threshold":
                FROZEN_MINIMUM_IMPROVEMENT,

            "runtime_selector_label_free":
                True,

            "annotations_used_at_runtime":
                False,

            "raw_image_handling":
                "byte_for_byte_copy",

            "processed_image_handling":
                "lossless_png",

            "label_handling":
                "byte_for_byte_copy",

            "geometry_changes_allowed":
                False,

            "threshold_recalibration_allowed":
                False,
        },
    )

    write_dataset_yaml(
        output_root
    )


# ============================================================
# CHECKPOINT / RESUME
# ============================================================


def load_progress(
    path: Path,
) -> dict[
    str,
    dict,
]:
    if not path.exists():
        return {}

    latest: dict[
        str,
        dict,
    ] = {}

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    for (
        index,
        line,
    ) in enumerate(
        lines
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(
                line
            )

        except json.JSONDecodeError:
            # A crash could leave only the very last line
            # partially written.
            if (
                index
                == len(
                    lines
                )
                - 1
            ):
                print(
                    "Warning: ignoring partial "
                    "final progress line."
                )

                break

            raise RuntimeError(
                "Corrupt progress manifest at line "
                f"{index + 1}: {path}"
            )

        source = str(
            record.get(
                "source_prepared_image",
                "",
            )
        )

        if not source:
            raise RuntimeError(
                "Progress manifest record has "
                "no source_prepared_image."
            )

        latest[
            source
        ] = record

    return latest


def append_progress(
    path: Path,
    record: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def checkpoint_valid(
    checkpoint: dict,
    source_record: dict,
    output_root: Path,
    policy_fingerprint: str,
) -> bool:
    if (
        checkpoint.get(
            "policy_fingerprint_sha256"
        )
        != policy_fingerprint
    ):
        raise RuntimeError(
            "Existing checkpoint was created "
            "using a different frozen policy. "
            "Use --reset for a clean run."
        )

    source_image = (
        PREPARED_ROOT
        / str(
            source_record[
                "prepared_image"
            ]
        )
    )

    source_label = (
        PREPARED_ROOT
        / str(
            source_record[
                "prepared_label"
            ]
        )
    )

    output_image_rel = str(
        checkpoint.get(
            "output_image",
            "",
        )
    )

    output_label_rel = str(
        checkpoint.get(
            "output_label",
            "",
        )
    )

    if (
        not output_image_rel
        or not output_label_rel
    ):
        return False

    output_image = (
        output_root
        / output_image_rel
    )

    output_label = (
        output_root
        / output_label_rel
    )

    for path in (
        source_image,
        source_label,
        output_image,
        output_label,
    ):
        if not path.is_file():
            return False

    if (
        sha256_file(
            source_image
        )
        != checkpoint.get(
            "source_image_sha256"
        )
    ):
        return False

    if (
        sha256_file(
            source_label
        )
        != checkpoint.get(
            "source_label_sha256"
        )
    ):
        return False

    if (
        sha256_file(
            output_image
        )
        != checkpoint.get(
            "output_image_sha256"
        )
    ):
        return False

    if (
        sha256_file(
            output_label
        )
        != checkpoint.get(
            "output_label_sha256"
        )
    ):
        return False

    if (
        checkpoint.get(
            "source_label_sha256"
        )
        != checkpoint.get(
            "output_label_sha256"
        )
    ):
        return False

    image = cv2.imread(
        str(
            output_image
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return False

    height, width = (
        image.shape[:2]
    )

    return (
        width
        == int(
            source_record[
                "width"
            ]
        )
        and height
        == int(
            source_record[
                "height"
            ]
        )
    )


# ============================================================
# OUTPUT IMAGE EXTENSION CLEANUP
# ============================================================


def remove_other_extensions(
    target: Path,
) -> None:
    for extension in (
        SUPPORTED_IMAGE_EXTENSIONS
    ):
        candidate = (
            target.parent
            / (
                f"{target.stem}"
                f"{extension}"
            )
        )

        if (
            candidate != target
            and candidate.exists()
        ):
            candidate.unlink()


# ============================================================
# PROCESS ONE IMAGE
# ============================================================


def process_one(
    record: dict,
    output_root: Path,
    policy: dict,
) -> dict:
    image_rel = str(
        record[
            "prepared_image"
        ]
    )

    label_rel = str(
        record[
            "prepared_label"
        ]
    )

    split = str(
        record[
            "split"
        ]
    )

    source_image = (
        PREPARED_ROOT
        / image_rel
    )

    source_label = (
        PREPARED_ROOT
        / label_rel
    )

    # --------------------------------------------------------
    # Source hashes
    # --------------------------------------------------------

    source_image_sha = (
        sha256_file(
            source_image
        )
    )

    source_label_sha = (
        sha256_file(
            source_label
        )
    )

    # --------------------------------------------------------
    # Optional Stage-1 source hash verification
    # --------------------------------------------------------

    metadata_sha = (
        record.get(
            "sha256"
        )
    )

    if (
        metadata_sha
        and str(
            metadata_sha
        )
        != source_image_sha
    ):
        raise RuntimeError(
            "Stage-1 prepared image hash mismatch: "
            f"{image_rel}"
        )

    # --------------------------------------------------------
    # Decode source image
    # --------------------------------------------------------

    raw = cv2.imread(
        str(
            source_image
        ),
        cv2.IMREAD_COLOR,
    )

    if raw is None:
        raise RuntimeError(
            "Cannot decode prepared image: "
            f"{source_image}"
        )

    height, width = (
        raw.shape[:2]
    )

    # --------------------------------------------------------
    # Geometry verification against Stage-1 metadata
    # --------------------------------------------------------

    if (
        width
        != int(
            record[
                "width"
            ]
        )
        or height
        != int(
            record[
                "height"
            ]
        )
    ):
        raise RuntimeError(
            "Prepared image geometry differs "
            "from Stage-1 metadata: "
            f"{image_rel}"
        )

    # --------------------------------------------------------
    # AUTHORITATIVE FROZEN SELECTOR
    #
    # The selector receives only image pixels.
    #
    # It receives:
    #     NO annotation boxes
    #     NO class labels
    #     NO target counts
    #
    # Threshold is fixed at 0.0200.
    # --------------------------------------------------------

    (
        winner_mode,
        winner_image,
        evaluations,
        _,
    ) = select_best_candidate(
        raw,
        FROZEN_MINIMUM_IMPROVEMENT,
    )

    if (
        winner_mode
        not in PREPROCESSING_MODES
    ):
        raise RuntimeError(
            "Selector returned unknown mode: "
            f"{winner_mode}"
        )

    # --------------------------------------------------------
    # Geometry is immutable
    # --------------------------------------------------------

    if (
        winner_image.shape
        != raw.shape
    ):
        raise RuntimeError(
            "Preprocessing changed image geometry: "
            f"{image_rel}"
        )

    processed = (
        winner_mode
        != "raw_baseline"
    )

    source_name = Path(
        image_rel
    )

    if processed:
        output_name = (
            source_name.stem
            + PROCESSED_IMAGE_EXTENSION
        )

    else:
        output_name = (
            source_name.name
        )

    output_image = (
        output_root
        / split
        / "images"
        / output_name
    )

    output_label = (
        output_root
        / split
        / "labels"
        / Path(
            label_rel
        ).name
    )

    # --------------------------------------------------------
    # Remove stale JPG/PNG alternatives if this image had
    # previously been generated differently.
    # --------------------------------------------------------

    remove_other_extensions(
        output_image
    )

    # --------------------------------------------------------
    # Write winner
    # --------------------------------------------------------

    if processed:
        atomic_write_png(
            winner_image,
            output_image,
        )

    else:
        atomic_copy(
            source_image,
            output_image,
        )

    # --------------------------------------------------------
    # Copy labels without modification
    # --------------------------------------------------------

    atomic_copy(
        source_label,
        output_label,
    )

    # --------------------------------------------------------
    # Output hashes
    # --------------------------------------------------------

    output_image_sha = (
        sha256_file(
            output_image
        )
    )

    output_label_sha = (
        sha256_file(
            output_label
        )
    )

    # --------------------------------------------------------
    # Label preservation
    # --------------------------------------------------------

    if (
        output_label_sha
        != source_label_sha
    ):
        raise RuntimeError(
            "YOLO label changed during "
            "master preprocessing: "
            f"{label_rel}"
        )

    # --------------------------------------------------------
    # Decode output and verify geometry
    # --------------------------------------------------------

    written = cv2.imread(
        str(
            output_image
        ),
        cv2.IMREAD_COLOR,
    )

    if written is None:
        raise RuntimeError(
            "Cannot decode written output image: "
            f"{output_image}"
        )

    if (
        written.shape
        != raw.shape
    ):
        raise RuntimeError(
            "Written output geometry differs "
            "from prepared image geometry: "
            f"{output_image}"
        )

    # --------------------------------------------------------
    # Processed PNG must reproduce exact selected pixels
    # --------------------------------------------------------

    if processed:
        if not np.array_equal(
            written,
            winner_image,
        ):
            raise RuntimeError(
                "Lossless PNG verification failed: "
                f"{output_image}"
            )

    # --------------------------------------------------------
    # Raw output must be exact source copy
    # --------------------------------------------------------

    else:
        if (
            output_image_sha
            != source_image_sha
        ):
            raise RuntimeError(
                "Raw fallback was not copied "
                "byte-for-byte: "
                f"{output_image}"
            )

    # --------------------------------------------------------
    # Winner evidence
    # --------------------------------------------------------

    winner_evaluation = (
        evaluations[
            winner_mode
        ]
    )

    if (
        winner_mode
        != "raw_baseline"
        and winner_evaluation.get(
            "damage_gate_passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Processed winner did not pass "
            "runtime safety gate: "
            f"{winner_mode}"
        )

    rejection_counts = Counter()

    for mode in PREPROCESSING_MODES:
        if mode == "raw_baseline":
            continue

        evaluation = (
            evaluations[
                mode
            ]
        )

        for reason in (
            evaluation.get(
                "rejection_reasons",
                [],
            )
        ):
            rejection_counts[
                str(
                    reason
                )
            ] += 1

    return {
        "source_prepared_image":
            image_rel,

        "source_prepared_label":
            label_rel,

        "output_image":
            relative_path(
                output_image,
                output_root,
            ),

        "output_label":
            relative_path(
                output_label,
                output_root,
            ),

        "split":
            split,

        "deployment":
            record.get(
                "deployment"
            ),

        "target_count":
            int(
                record.get(
                    "target_count",
                    0,
                )
            ),

        "width":
            width,

        "height":
            height,

        "selected_mode":
            winner_mode,

        "processed":
            processed,

        "winner_damage_gate_passed":
            bool(
                winner_evaluation.get(
                    "damage_gate_passed",
                    True,
                )
            ),

        "winner_benefit_score":
            float(
                winner_evaluation.get(
                    "benefit_score",
                    0.0,
                )
            ),

        "winner_safety_margin":
            float(
                winner_evaluation.get(
                    "safety_margin_score",
                    1.0,
                )
            ),

        "candidate_rejection_reason_counts":
            dict(
                rejection_counts
            ),

        "source_image_sha256":
            source_image_sha,

        "source_label_sha256":
            source_label_sha,

        "output_image_sha256":
            output_image_sha,

        "output_label_sha256":
            output_label_sha,

        "geometry_preserved":
            True,

        "label_preserved_byte_for_byte":
            True,

        "raw_preserved_byte_for_byte":
            (
                not processed
            ),

        "processed_png_lossless":
            processed,

        "policy_version":
            FROZEN_POLICY_VERSION,

        "selector_source_version":
            FROZEN_SELECTOR_SOURCE_VERSION,

        "minimum_improvement":
            FROZEN_MINIMUM_IMPROVEMENT,

        "policy_fingerprint_sha256":
            policy[
                "policy_fingerprint_sha256"
            ],

        "completed_at_utc":
            utc_now(),
    }


# ============================================================
# FINAL MANIFEST WRITING
# ============================================================


def write_manifests(
    output_root: Path,
    records: list[dict],
) -> None:
    records = sorted(
        records,
        key=lambda item:
            item[
                "source_prepared_image"
            ],
    )

    manifests_root = (
        output_root
        / "manifests"
    )

    # --------------------------------------------------------
    # JSONL manifest
    # --------------------------------------------------------

    jsonl_path = (
        manifests_root
        / "preprocessing_manifest.jsonl"
    )

    temporary_jsonl = (
        jsonl_path.with_name(
            jsonl_path.name
            + ".tmp"
        )
    )

    with temporary_jsonl.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        temporary_jsonl,
        jsonl_path,
    )

    # --------------------------------------------------------
    # CSV manifest
    # --------------------------------------------------------

    fields = [
        "source_prepared_image",
        "output_image",
        "split",
        "deployment",
        "target_count",
        "selected_mode",
        "processed",
        "winner_benefit_score",
        "winner_safety_margin",
        "source_image_sha256",
        "output_image_sha256",
        "source_label_sha256",
        "output_label_sha256",
    ]

    csv_path = (
        manifests_root
        / "preprocessing_manifest.csv"
    )

    temporary_csv = (
        csv_path.with_name(
            csv_path.name
            + ".tmp"
        )
    )

    with temporary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    field:
                        record.get(
                            field
                        )
                    for field
                    in fields
                }
            )

    os.replace(
        temporary_csv,
        csv_path,
    )

    # --------------------------------------------------------
    # Final image metadata index
    # --------------------------------------------------------

    write_json(
        output_root
        / "metadata"
        / "image_index.json",

        records,
    )


# ============================================================
# FINAL DATASET VERIFICATION
# ============================================================


def final_verify(
    source_records: list[dict],
    records: list[dict],
    output_root: Path,
) -> dict:
    print_header(
        "FINAL OUTPUT VERIFICATION"
    )

    expected_sources = {
        str(
            record[
                "prepared_image"
            ]
        )
        for record
        in source_records
    }

    actual_sources = {
        str(
            record[
                "source_prepared_image"
            ]
        )
        for record
        in records
    }

    if (
        expected_sources
        != actual_sources
    ):
        raise RuntimeError(
            "Final preprocessing manifest does not "
            "exactly match selected prepared inputs."
        )

    split_results = {}

    # --------------------------------------------------------
    # Exact directory-content verification
    # --------------------------------------------------------

    for split in (
        "train",
        "val",
        "test",
    ):
        split_records = [
            record
            for record
            in records
            if record[
                "split"
            ]
            == split
        ]

        expected_images = {
            Path(
                record[
                    "output_image"
                ]
            ).name
            for record
            in split_records
        }

        expected_labels = {
            Path(
                record[
                    "output_label"
                ]
            ).name
            for record
            in split_records
        }

        image_directory = (
            output_root
            / split
            / "images"
        )

        label_directory = (
            output_root
            / split
            / "labels"
        )

        actual_images = {
            path.name
            for path
            in image_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
            )
        }

        actual_labels = {
            path.name
            for path
            in label_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                == ".txt"
            )
        }

        if (
            actual_images
            != expected_images
        ):
            missing = sorted(
                expected_images
                - actual_images
            )[:10]

            extra = sorted(
                actual_images
                - expected_images
            )[:10]

            raise RuntimeError(
                f"Image-set mismatch in {split}. "
                f"Missing={missing}; extra={extra}"
            )

        if (
            actual_labels
            != expected_labels
        ):
            missing = sorted(
                expected_labels
                - actual_labels
            )[:10]

            extra = sorted(
                actual_labels
                - expected_labels
            )[:10]

            raise RuntimeError(
                f"Label-set mismatch in {split}. "
                f"Missing={missing}; extra={extra}"
            )

        split_results[
            split
        ] = len(
            split_records
        )

    # --------------------------------------------------------
    # Hash + geometry verification for all final outputs
    # --------------------------------------------------------

    for record in tqdm(
        records,
        desc="Verifying outputs",
        unit="img",
        dynamic_ncols=True,
    ):
        output_image = (
            output_root
            / record[
                "output_image"
            ]
        )

        output_label = (
            output_root
            / record[
                "output_label"
            ]
        )

        if (
            sha256_file(
                output_image
            )
            != record[
                "output_image_sha256"
            ]
        ):
            raise RuntimeError(
                "Final output image hash mismatch: "
                f"{output_image}"
            )

        if (
            sha256_file(
                output_label
            )
            != record[
                "output_label_sha256"
            ]
        ):
            raise RuntimeError(
                "Final output label hash mismatch: "
                f"{output_label}"
            )

        if (
            record[
                "source_label_sha256"
            ]
            != record[
                "output_label_sha256"
            ]
        ):
            raise RuntimeError(
                "Label preservation failure: "
                f"{output_label}"
            )

        image = cv2.imread(
            str(
                output_image
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                "Cannot decode final output image: "
                f"{output_image}"
            )

        height, width = (
            image.shape[:2]
        )

        if (
            width
            != int(
                record[
                    "width"
                ]
            )
            or height
            != int(
                record[
                    "height"
                ]
            )
        ):
            raise RuntimeError(
                "Geometry changed in final output: "
                f"{output_image}"
            )

    print(
        f"Images verified           : "
        f"{len(records):,}"
    )

    print(
        f"Labels verified           : "
        f"{len(records):,}"
    )

    print(
        "Label mismatches          : 0"
    )

    print(
        "Geometry failures         : 0"
    )

    print(
        "Hash failures             : 0"
    )

    print(
        "Result                    : PASS"
    )

    return {
        "passed":
            True,

        "record_count":
            len(
                records
            ),

        "split_counts":
            split_results,

        "label_mismatches":
            0,

        "geometry_failures":
            0,

        "hash_failures":
            0,
    }


# ============================================================
# MASTER SMOKE SAMPLE SELECTION
# ============================================================


def evenly_spaced(
    records: list[dict],
    count: int,
) -> list[dict]:
    records = sorted(
        records,
        key=lambda item:
            item[
                "prepared_image"
            ],
    )

    if count >= len(
        records
    ):
        return records

    if count == 1:
        return [
            records[
                len(
                    records
                )
                // 2
            ]
        ]

    return [
        records[
            round(
                index
                * (
                    len(
                        records
                    )
                    - 1
                )
                / (
                    count
                    - 1
                )
            )
        ]
        for index
        in range(
            count
        )
    ]


def select_master_smoke_records(
    image_index: list[dict],
    per_split: int,
    certified_smoke_report: dict,
) -> list[dict]:
    by_path = {
        record[
            "prepared_image"
        ]:
            record
        for record
        in image_index
    }

    known_processed = None

    # --------------------------------------------------------
    # Guarantee that the isolated master smoke test exercises
    # at least one known processed-winner image.
    # --------------------------------------------------------

    decisions = (
        certified_smoke_report.get(
            "decisions",
            []
        )
    )

    if not isinstance(
        decisions,
        list,
    ):
        decisions = []

    for decision in decisions:
        if not isinstance(
            decision,
            dict,
        ):
            continue

        if not decision.get(
            "processed_selected"
        ):
            continue

        candidate = by_path.get(
            decision.get(
                "prepared_image"
            )
        )

        if (
            candidate is not None
            and candidate.get(
                "split"
            )
            == "train"
        ):
            known_processed = (
                candidate
            )

            break

    selected = []

    for split in (
        "train",
        "val",
        "test",
    ):
        pool = [
            record
            for record
            in image_index
            if record.get(
                "split"
            )
            == split
        ]

        picks = evenly_spaced(
            pool,
            per_split,
        )

        if (
            split == "train"
            and known_processed is not None
            and picks
        ):
            if all(
                item[
                    "prepared_image"
                ]
                != known_processed[
                    "prepared_image"
                ]
                for item
                in picks
            ):
                picks[
                    -1
                ] = known_processed

        selected.extend(
            picks
        )

    deduplicated = {
        record[
            "prepared_image"
        ]:
            record
        for record
        in selected
    }

    result = list(
        deduplicated.values()
    )

    return sorted(
        result,
        key=lambda record: (
            record[
                "split"
            ],
            record[
                "prepared_image"
            ],
        ),
    )


# ============================================================
# MASTER RUNNER
# ============================================================


def run(
    source_records: list[dict],
    dataset_info: dict,
    certified: dict,
    output_root: Path,
    run_mode: str,
    reset: bool,
) -> None:
    started_at_utc = (
        utc_now()
    )

    started = (
        time.perf_counter()
    )

    prepare_output_root(
        output_root,
        reset,
    )

    policy = (
        certified[
            "policy"
        ]
    )

    write_initial_metadata(
        output_root,
        policy,
        dataset_info,
        run_mode,
    )

    progress_file = (
        output_root
        / "manifests"
        / "progress.jsonl"
    )

    progress = load_progress(
        progress_file
    )

    final_records: dict[
        str,
        dict,
    ] = {}

    resumed_count = 0
    processed_now_count = 0

    try:
        # ----------------------------------------------------
        # Main processing loop
        # ----------------------------------------------------

        for source_record in tqdm(
            source_records,
            desc="Master preprocessing",
            unit="img",
            dynamic_ncols=True,
        ):
            key = str(
                source_record[
                    "prepared_image"
                ]
            )

            checkpoint = (
                progress.get(
                    key
                )
            )

            if (
                checkpoint
                and checkpoint_valid(
                    checkpoint,
                    source_record,
                    output_root,
                    policy[
                        "policy_fingerprint_sha256"
                    ],
                )
            ):
                final_records[
                    key
                ] = checkpoint

                resumed_count += 1

                continue

            completed = process_one(
                source_record,
                output_root,
                policy,
            )

            append_progress(
                progress_file,
                completed,
            )

            final_records[
                key
            ] = completed

            processed_now_count += 1

        # ----------------------------------------------------
        # Deterministic manifest order
        # ----------------------------------------------------

        ordered_records = [
            final_records[
                record[
                    "prepared_image"
                ]
            ]
            for record
            in source_records
        ]

        # ----------------------------------------------------
        # Manifests
        # ----------------------------------------------------

        write_manifests(
            output_root,
            ordered_records,
        )

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------

        verification = (
            final_verify(
                source_records,
                ordered_records,
                output_root,
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mode_counts = Counter(
            record[
                "selected_mode"
            ]
            for record
            in ordered_records
        )

        rejection_counts = (
            Counter()
        )

        split_mode_counts: dict[
            str,
            Counter,
        ] = {
            "train":
                Counter(),

            "val":
                Counter(),

            "test":
                Counter(),
        }

        for record in ordered_records:
            split_mode_counts[
                record[
                    "split"
                ]
            ][
                record[
                    "selected_mode"
                ]
            ] += 1

            for (
                reason,
                count,
            ) in record[
                "candidate_rejection_reason_counts"
            ].items():
                rejection_counts[
                    reason
                ] += int(
                    count
                )

        raw_count = (
            mode_counts.get(
                "raw_baseline",
                0,
            )
        )

        processed_count = (
            len(
                ordered_records
            )
            - raw_count
        )

        duration_seconds = (
            time.perf_counter()
            - started
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report = {
            "status":
                (
                    "CERTIFIED"
                    if run_mode
                    == "full"
                    else "SMOKE_TEST_PASS"
                ),

            "generated_at_utc":
                utc_now(),

            "started_at_utc":
                started_at_utc,

            "duration_seconds":
                round(
                    duration_seconds,
                    3,
                ),

            "run_mode":
                run_mode,

            "policy_version":
                FROZEN_POLICY_VERSION,

            "selector_source_version":
                FROZEN_SELECTOR_SOURCE_VERSION,

            "policy_fingerprint_sha256":
                policy[
                    "policy_fingerprint_sha256"
                ],

            "frozen_threshold":
                FROZEN_MINIMUM_IMPROVEMENT,

            "runtime_selector_label_free":
                True,

            "annotations_used_at_runtime":
                False,

            "record_count":
                len(
                    ordered_records
                ),

            "resumed_count":
                resumed_count,

            "processed_now_count":
                processed_now_count,

            "raw_winners":
                raw_count,

            "processed_winners":
                processed_count,

            "processed_rate":
                (
                    processed_count
                    / len(
                        ordered_records
                    )
                    if ordered_records
                    else 0.0
                ),

            "winner_distribution": {
                mode:
                    mode_counts.get(
                        mode,
                        0,
                    )
                for mode
                in PREPROCESSING_MODES
            },

            "winner_distribution_by_split": {
                split: {
                    mode:
                        split_mode_counts[
                            split
                        ].get(
                            mode,
                            0,
                        )
                    for mode
                    in PREPROCESSING_MODES
                }
                for split
                in (
                    "train",
                    "val",
                    "test",
                )
            },

            "candidate_rejection_reason_counts":
                dict(
                    rejection_counts
                ),

            "verification":
                verification,

            "output_root":
                str(
                    output_root
                ),
        }

        write_json(
            output_root
            / "reports"
            / "master_preprocessing_report.json",

            report,
        )

        write_json(
            output_root
            / "reports"
            / "certification_report.json",

            {
                "status":
                    report[
                        "status"
                    ],

                "policy_version":
                    FROZEN_POLICY_VERSION,

                "selector_source_version":
                    FROZEN_SELECTOR_SOURCE_VERSION,

                "frozen_threshold":
                    FROZEN_MINIMUM_IMPROVEMENT,

                "policy_fingerprint_sha256":
                    policy[
                        "policy_fingerprint_sha256"
                    ],

                "input_output_count_match":
                    True,

                "geometry_failures":
                    0,

                "label_mismatches":
                    0,

                "hash_failures":
                    0,

                "runtime_annotations_used":
                    False,

                "threshold_recalibrated":
                    False,

                "generated_at_utc":
                    utc_now(),
            },
        )

        # ----------------------------------------------------
        # Human-readable summary
        # ----------------------------------------------------

        print_header(
            "MASTER PREPROCESSING SUMMARY"
        )

        print(
            f"Status                    : "
            f"{report['status']}"
        )

        print(
            f"Run mode                  : "
            f"{run_mode}"
        )

        print(
            f"Frozen threshold          : "
            f"{FROZEN_MINIMUM_IMPROVEMENT:.4f}"
        )

        print(
            f"Images                    : "
            f"{len(ordered_records):,}"
        )

        print(
            f"Resumed                   : "
            f"{resumed_count:,}"
        )

        print(
            f"Processed this run        : "
            f"{processed_now_count:,}"
        )

        print(
            f"Raw winners               : "
            f"{raw_count:,}"
        )

        print(
            f"Processed winners         : "
            f"{processed_count:,}"
        )

        print(
            f"Processed rate            : "
            f"{report['processed_rate'] * 100:.2f}%"
        )

        print()

        print(
            "Winner distribution:"
        )

        for mode in PREPROCESSING_MODES:
            count = (
                mode_counts.get(
                    mode,
                    0,
                )
            )

            percentage = (
                count
                / len(
                    ordered_records
                )
                * 100.0
                if ordered_records
                else 0.0
            )

            print(
                f"  {mode:<30}"
                f"{count:>6,} "
                f"({percentage:>6.2f}%)"
            )

        print()

        print(
            "Dataset verification      : PASS"
        )

        print(
            "Geometry changes           : 0"
        )

        print(
            "Label changes              : 0"
        )

        print(
            "Hash failures              : 0"
        )

        print(
            f"Output root               : "
            f"{output_root}"
        )

    except Exception as exc:
        write_json(
            output_root
            / "reports"
            / "failure_report.json",

            {
                "status":
                    "FAILED",

                "generated_at_utc":
                    utc_now(),

                "run_mode":
                    run_mode,

                "policy_version":
                    FROZEN_POLICY_VERSION,

                "processed_now_count":
                    processed_now_count,

                "resumed_count":
                    resumed_count,

                "exception_type":
                    type(
                        exc
                    ).__name__,

                "exception_message":
                    str(
                        exc
                    ),
            },
        )

        raise


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen underwater-fish "
            "preprocessing policy v1.4.3."
        )
    )

    action_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    action_group.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate frozen policy and prepared "
            "dataset without preprocessing images."
        ),
    )

    action_group.add_argument(
        "--smoke-test",
        type=int,
        metavar="N",
        help=(
            "Process N deterministic images per split "
            "into an isolated master smoke-test folder."
        ),
    )

    action_group.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run master preprocessing across the "
            "complete prepared dataset."
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete the selected output root before "
            "running. Recommended only for a clean "
            "first full run."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    print_header(
        "FISH DETECTION - "
        "MASTER PREPROCESSING v1.4.3"
    )

    print(
        f"Frozen threshold          : "
        f"{FROZEN_MINIMUM_IMPROVEMENT:.4f}"
    )

    print(
        "Runtime selector          : LABEL-FREE"
    )

    print(
        "Annotations at runtime    : NO"
    )

    print(
        "Threshold recalibration   : DISABLED"
    )

    print(
        "Raw output                : "
        "BYTE-FOR-BYTE COPY"
    )

    print(
        "Processed output          : "
        "LOSSLESS PNG"
    )

    print(
        "Labels                    : "
        "BYTE-FOR-BYTE COPY"
    )

    image_index, dataset_info = (
        load_source_records()
    )

    certified = preflight(
        image_index,
        dataset_info,
    )

    # --------------------------------------------------------
    # PREFLIGHT ONLY
    # --------------------------------------------------------

    if args.preflight:
        print()

        print(
            "Preflight passed."
        )

        print(
            "No preprocessed dataset outputs "
            "were generated."
        )

        return

    # --------------------------------------------------------
    # MASTER SMOKE TEST
    # --------------------------------------------------------

    if args.smoke_test is not None:
        if args.smoke_test < 1:
            raise ValueError(
                "--smoke-test must be >= 1."
            )

        source_records = (
            select_master_smoke_records(
                image_index,
                args.smoke_test,
                certified[
                    "smoke_report"
                ],
            )
        )

        output_root = (
            MASTER_SMOKE_ROOT
        )

        run_mode = "smoke"

        # Smoke test is isolated and always starts clean.
        reset = True

        print_header(
            "MASTER PREPROCESSING SMOKE RUN"
        )

        print(
            f"Requested per split       : "
            f"{args.smoke_test}"
        )

        print(
            f"Selected records          : "
            f"{len(source_records)}"
        )

        print(
            f"Output root               : "
            f"{output_root}"
        )

    # --------------------------------------------------------
    # COMPLETE DATASET
    # --------------------------------------------------------

    else:
        source_records = (
            image_index
        )

        output_root = (
            PREPROCESSED_ROOT
        )

        run_mode = "full"

        reset = (
            args.reset
        )

        print_header(
            "FULL MASTER PREPROCESSING RUN"
        )

        print(
            f"Prepared records          : "
            f"{len(source_records):,}"
        )

        print(
            f"Resume enabled            : "
            f"{not reset}"
        )

        print(
            f"Clean reset               : "
            f"{reset}"
        )

        print(
            f"Output root               : "
            f"{output_root}"
        )

    run(
        source_records=source_records,
        dataset_info=dataset_info,
        certified=certified,
        output_root=output_root,
        run_mode=run_mode,
        reset=reset,
    )


if __name__ == "__main__":
    main()