from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import requests
import yaml
from tqdm import tqdm

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_ROOT = PROJECT_ROOT / "master_dataset"

DOWNLOAD_ROOT = MASTER_ROOT / "downloads"
RAW_ROOT = MASTER_ROOT / "raw" / "brackish"

MASTER_SPLITS_ROOT = MASTER_ROOT / "splits"
MASTER_METADATA_ROOT = MASTER_ROOT / "metadata"
MASTER_REPORTS_ROOT = MASTER_ROOT / "reports"

PREPARED_ROOT = (
    PROJECT_ROOT
    / "preprocessing"
    / "prepared_dataset"
)

PREPARED_MANIFESTS_ROOT = PREPARED_ROOT / "manifests"
PREPARED_METADATA_ROOT = PREPARED_ROOT / "metadata"
PREPARED_REPORTS_ROOT = PREPARED_ROOT / "reports"

DATASET_YAML = PREPARED_ROOT / "dataset.yaml"

SOURCE_ZIP = DOWNLOAD_ROOT / "brackish.zip"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

PREPARATION_VERSION = "1.0"

SOURCE_URL = (
    "https://public.roboflow.com/ds/"
    "vGBLxigwno?key=bhFPGoB3VB"
)

SOURCE_SPLITS = (
    "train",
    "valid",
    "test",
)

TARGET_CATEGORY_NAMES = {
    "fish",
    "small_fish",
}

FINAL_CLASS_ID = 0
FINAL_CLASS_NAME = "fish"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SPLIT_NAMES = (
    "train",
    "val",
    "test",
)

SPLIT_SEED = 42

# More attempts = better chance of balancing deployment groups.
SPLIT_SEARCH_ATTEMPTS = 300

# Hardlinks save disk space because source/prepared images are
# on the same drive. If hardlinking fails, copy is used.
USE_HARDLINKS = True


# ============================================================
# GENERAL UTILITIES
# ============================================================


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def write_json(
    path: Path,
    data: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def normalize_relative_path(
    path_value: str,
) -> str:
    return PurePosixPath(
        path_value.replace(
            "\\",
            "/",
        )
    ).as_posix()


def source_path_from_record(
    record: dict,
) -> Path:
    return (
        RAW_ROOT
        / record["source_split"]
        / Path(
            *PurePosixPath(
                record["file_name"]
            ).parts
        )
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# RAW DATASET ACQUISITION
# ============================================================


def source_structure_valid(
    root: Path,
) -> bool:
    for split in SOURCE_SPLITS:
        split_dir = root / split

        annotation = (
            split_dir
            / "_annotations.coco.json"
        )

        if (
            not split_dir.is_dir()
            or not annotation.is_file()
        ):
            return False

    return True


def download_dataset() -> None:
    DOWNLOAD_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        SOURCE_ZIP.exists()
        and zipfile.is_zipfile(
            SOURCE_ZIP
        )
    ):
        print(
            f"[DOWNLOAD] Existing ZIP found: "
            f"{SOURCE_ZIP}"
        )

        return

    if SOURCE_ZIP.exists():
        SOURCE_ZIP.unlink()

    temporary_file = (
        SOURCE_ZIP.with_suffix(
            ".zip.part"
        )
    )

    if temporary_file.exists():
        temporary_file.unlink()

    print(
        "[DOWNLOAD] Downloading Brackish dataset..."
    )

    with requests.get(
        SOURCE_URL,
        stream=True,
        timeout=(30, 120),
    ) as response:

        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        with (
            temporary_file.open(
                "wb"
            ) as output,
            tqdm(
                total=(
                    total_size
                    if total_size > 0
                    else None
                ),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Brackish download",
                dynamic_ncols=True,
            ) as progress,
        ):
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                output.write(
                    chunk
                )

                progress.update(
                    len(chunk)
                )

    temporary_file.replace(
        SOURCE_ZIP
    )

    if not zipfile.is_zipfile(
        SOURCE_ZIP
    ):
        SOURCE_ZIP.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded Brackish archive is not a valid ZIP."
        )

    print(
        "[DOWNLOAD] Download complete."
    )


def find_dataset_root(
    extraction_root: Path,
) -> Path | None:
    candidates = [
        extraction_root
    ]

    candidates.extend(
        path
        for path in extraction_root.rglob("*")
        if path.is_dir()
    )

    for candidate in candidates:
        if source_structure_valid(
            candidate
        ):
            return candidate

    return None


def extract_dataset() -> None:
    extraction_root = (
        DOWNLOAD_ROOT
        / "_brackish_extract_tmp"
    )

    if extraction_root.exists():
        shutil.rmtree(
            extraction_root
        )

    extraction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[EXTRACT] Extracting Brackish dataset..."
    )

    with zipfile.ZipFile(
        SOURCE_ZIP,
        "r",
    ) as archive:
        archive.extractall(
            extraction_root
        )

    discovered_root = (
        find_dataset_root(
            extraction_root
        )
    )

    if discovered_root is None:
        shutil.rmtree(
            extraction_root,
            ignore_errors=True,
        )

        raise RuntimeError(
            "Could not locate train/valid/test COCO "
            "folders inside the downloaded archive."
        )

    # Only remove RAW_ROOT if it does not already contain
    # a valid source dataset.
    if RAW_ROOT.exists():
        if source_structure_valid(
            RAW_ROOT
        ):
            shutil.rmtree(
                extraction_root,
                ignore_errors=True,
            )

            return

        remaining = [
            item
            for item in RAW_ROOT.iterdir()
            if item.name != ".gitkeep"
        ]

        if remaining:
            raise RuntimeError(
                "master_dataset/raw/brackish contains "
                "unexpected files but is not a valid dataset. "
                "Refusing to overwrite it automatically."
            )

        shutil.rmtree(
            RAW_ROOT
        )

    RAW_ROOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.move(
        str(discovered_root),
        str(RAW_ROOT),
    )

    if extraction_root.exists():
        shutil.rmtree(
            extraction_root,
            ignore_errors=True,
        )

    if not source_structure_valid(
        RAW_ROOT
    ):
        raise RuntimeError(
            "Dataset extraction completed but expected "
            "Brackish structure was not found."
        )

    print(
        "[EXTRACT] Dataset ready."
    )


def ensure_source_dataset() -> None:
    print_header(
        "SOURCE DATASET"
    )

    if source_structure_valid(
        RAW_ROOT
    ):
        print(
            "[OK] Existing Brackish dataset found."
        )

        print(
            RAW_ROOT
        )

        return

    print(
        "[INFO] Brackish source dataset not found."
    )

    download_dataset()

    extract_dataset()

    print(
        f"[OK] Source dataset: {RAW_ROOT}"
    )


# ============================================================
# DEPLOYMENT / VIDEO GROUP IDENTIFICATION
# ============================================================


def get_deployment_name(
    file_name: str,
) -> str:
    """
    Brackish images are video frames.

    The final frame suffix is removed so frames from the same
    video/deployment remain in the same train/val/test split.
    """

    base = Path(
        file_name
    ).stem

    # Remove Roboflow-generated hash suffix.
    if ".rf." in base:
        base = base.split(
            ".rf.",
            1,
        )[0]

    for suffix in (
        "_jpg",
        "_jpeg",
        "_png",
    ):
        if suffix in base:
            base = base.split(
                suffix,
                1,
            )[0]

    dash_parts = base.split(
        "-"
    )

    if len(dash_parts) > 1:
        return "-".join(
            dash_parts[:-1]
        )

    underscore_parts = base.split(
        "_"
    )

    if (
        len(underscore_parts) > 1
        and re.fullmatch(
            r"\d+",
            underscore_parts[-1],
        )
    ):
        return "_".join(
            underscore_parts[:-1]
        )

    return base


# ============================================================
# COCO / SOURCE AUDIT
# ============================================================


def valid_number(value: object) -> bool:
    return (
        isinstance(
            value,
            (int, float),
        )
        and math.isfinite(
            float(value)
        )
    )


def validate_bbox(
    bbox: object,
    width: int,
    height: int,
) -> tuple[
    bool,
    str | None,
]:
    if (
        not isinstance(
            bbox,
            list,
        )
        or len(bbox) != 4
    ):
        return (
            False,
            "bbox_not_xywh",
        )

    if not all(
        valid_number(value)
        for value in bbox
    ):
        return (
            False,
            "bbox_non_numeric",
        )

    x, y, box_width, box_height = map(
        float,
        bbox,
    )

    if (
        box_width <= 0
        or box_height <= 0
    ):
        return (
            False,
            "bbox_non_positive_size",
        )

    if x < 0 or y < 0:
        return (
            False,
            "bbox_negative_origin",
        )

    epsilon = 1e-3

    if (
        x + box_width
        > width + epsilon
    ):
        return (
            False,
            "bbox_exceeds_width",
        )

    if (
        y + box_height
        > height + epsilon
    ):
        return (
            False,
            "bbox_exceeds_height",
        )

    return (
        True,
        None,
    )


def audit_source_dataset() -> tuple[
    list[dict],
    dict,
]:
    print_header(
        "SOURCE DATASET AUDIT"
    )

    issues = {
        "missing_images": [],
        "corrupt_images": [],
        "orphan_images": [],
        "orphan_annotations": [],
        "invalid_boxes": [],
        "dimension_mismatches": [],
        "unknown_categories": [],
        "missing_target_categories": [],
    }

    records: list[dict] = []

    category_counts: Counter = Counter()

    source_split_counts: dict[
        str,
        int,
    ] = {}

    for source_split in SOURCE_SPLITS:

        split_dir = (
            RAW_ROOT
            / source_split
        )

        annotation_path = (
            split_dir
            / "_annotations.coco.json"
        )

        coco = json.loads(
            annotation_path.read_text(
                encoding="utf-8"
            )
        )

        categories = {
            int(category["id"]):
                str(
                    category["name"]
                ).strip()
            for category in coco.get(
                "categories",
                [],
            )
        }

        normalized_categories = {
            category_id:
                category_name.lower()
            for (
                category_id,
                category_name,
            ) in categories.items()
        }

        available_target_names = {
            name
            for name
            in normalized_categories.values()
            if name
            in TARGET_CATEGORY_NAMES
        }

        missing_targets = (
            TARGET_CATEGORY_NAMES
            - available_target_names
        )

        if missing_targets:
            issues[
                "missing_target_categories"
            ].append(
                {
                    "source_split":
                        source_split,
                    "missing":
                        sorted(
                            missing_targets
                        ),
                }
            )

        images = {
            int(image["id"]):
                image
            for image
            in coco.get(
                "images",
                [],
            )
        }

        annotations_by_image: dict[
            int,
            list[dict],
        ] = defaultdict(
            list
        )

        for annotation in coco.get(
            "annotations",
            [],
        ):
            image_id = annotation.get(
                "image_id"
            )

            category_id = annotation.get(
                "category_id"
            )

            if (
                not isinstance(
                    image_id,
                    int,
                )
                or image_id
                not in images
            ):
                issues[
                    "orphan_annotations"
                ].append(
                    {
                        "source_split":
                            source_split,
                        "annotation_id":
                            annotation.get(
                                "id"
                            ),
                        "image_id":
                            image_id,
                    }
                )

                continue

            if category_id not in categories:
                issues[
                    "unknown_categories"
                ].append(
                    {
                        "source_split":
                            source_split,
                        "annotation_id":
                            annotation.get(
                                "id"
                            ),
                        "category_id":
                            category_id,
                    }
                )

                continue

            category_counts[
                normalized_categories[
                    category_id
                ]
            ] += 1

            annotations_by_image[
                image_id
            ].append(
                annotation
            )

        listed_image_paths: set[
            str
        ] = set()

        split_records = 0

        progress = tqdm(
            images.values(),
            desc=f"Auditing {source_split}",
            unit="img",
            dynamic_ncols=True,
        )

        for image_metadata in progress:
            image_id = int(
                image_metadata["id"]
            )

            file_name = (
                normalize_relative_path(
                    str(
                        image_metadata[
                            "file_name"
                        ]
                    )
                )
            )

            listed_image_paths.add(
                file_name
            )

            source_path = (
                split_dir
                / Path(
                    *PurePosixPath(
                        file_name
                    ).parts
                )
            )

            source_relative = (
                PurePosixPath(
                    source_split
                )
                / PurePosixPath(
                    file_name
                )
            ).as_posix()

            if not source_path.exists():
                issues[
                    "missing_images"
                ].append(
                    source_relative
                )

                continue

            image = cv2.imread(
                str(source_path)
            )

            if image is None:
                issues[
                    "corrupt_images"
                ].append(
                    source_relative
                )

                continue

            actual_height, actual_width = (
                image.shape[:2]
            )

            coco_width = image_metadata.get(
                "width"
            )

            coco_height = image_metadata.get(
                "height"
            )

            if (
                coco_width != actual_width
                or coco_height
                != actual_height
            ):
                issues[
                    "dimension_mismatches"
                ].append(
                    {
                        "image":
                            source_relative,
                        "coco_width":
                            coco_width,
                        "coco_height":
                            coco_height,
                        "actual_width":
                            actual_width,
                        "actual_height":
                            actual_height,
                    }
                )

            target_boxes: list[
                dict
            ] = []

            target_counts = Counter()

            non_target_count = 0

            for annotation in (
                annotations_by_image.get(
                    image_id,
                    [],
                )
            ):
                category_id = int(
                    annotation[
                        "category_id"
                    ]
                )

                category_name = (
                    normalized_categories[
                        category_id
                    ]
                )

                bbox = annotation.get(
                    "bbox"
                )

                (
                    bbox_valid,
                    bbox_error,
                ) = validate_bbox(
                    bbox,
                    actual_width,
                    actual_height,
                )

                if not bbox_valid:
                    issues[
                        "invalid_boxes"
                    ].append(
                        {
                            "image":
                                source_relative,
                            "annotation_id":
                                annotation.get(
                                    "id"
                                ),
                            "category":
                                category_name,
                            "bbox":
                                bbox,
                            "reason":
                                bbox_error,
                        }
                    )

                    continue

                if (
                    category_name
                    in TARGET_CATEGORY_NAMES
                ):
                    target_counts[
                        category_name
                    ] += 1

                    target_boxes.append(
                        {
                            "source_annotation_id":
                                annotation.get(
                                    "id"
                                ),
                            "source_category":
                                category_name,
                            "bbox_xywh": [
                                float(value)
                                for value
                                in bbox
                            ],
                        }
                    )

                else:
                    non_target_count += 1

            record = {
                "source_split":
                    source_split,

                "source_image_id":
                    image_id,

                "file_name":
                    file_name,

                "source_relative_path":
                    source_relative,

                "width":
                    actual_width,

                "height":
                    actual_height,

                "deployment":
                    get_deployment_name(
                        file_name
                    ),

                "sha256":
                    sha256_file(
                        source_path
                    ),

                "fish_count":
                    target_counts[
                        "fish"
                    ],

                "small_fish_count":
                    target_counts[
                        "small_fish"
                    ],

                "target_count":
                    len(
                        target_boxes
                    ),

                "non_target_count":
                    non_target_count,

                "target_boxes":
                    target_boxes,

                "duplicate_sources": [],
            }

            records.append(
                record
            )

            split_records += 1

        source_split_counts[
            source_split
        ] = split_records

        actual_image_paths = {
            path.relative_to(
                split_dir
            ).as_posix()
            for path
            in split_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        }

        orphan_images = sorted(
            actual_image_paths
            - listed_image_paths
        )

        for orphan in orphan_images:
            issues[
                "orphan_images"
            ].append(
                (
                    PurePosixPath(
                        source_split
                    )
                    / orphan
                ).as_posix()
            )

    audit = {
        "prepared_at_utc":
            utc_now(),

        "source_path":
            str(
                RAW_ROOT
            ),

        "source_split_counts":
            source_split_counts,

        "usable_records":
            len(
                records
            ),

        "source_category_annotation_counts":
            dict(
                sorted(
                    category_counts.items()
                )
            ),

        "issues":
            issues,
    }

    return (
        records,
        audit,
    )


# ============================================================
# SOURCE AUDIT CERTIFICATION
# ============================================================


def source_fatal_issue_count(
    audit: dict,
) -> int:
    fatal_keys = (
        "missing_images",
        "corrupt_images",
        "orphan_images",
        "orphan_annotations",
        "invalid_boxes",
        "dimension_mismatches",
        "unknown_categories",
        "missing_target_categories",
    )

    return sum(
        len(
            audit[
                "issues"
            ][key]
        )
        for key
        in fatal_keys
    )


# ============================================================
# EXACT DUPLICATE DETECTION
# ============================================================


def target_signature(
    record: dict,
) -> tuple:
    boxes = []

    for annotation in record[
        "target_boxes"
    ]:
        bbox = annotation[
            "bbox_xywh"
        ]

        boxes.append(
            tuple(
                round(
                    float(value),
                    4,
                )
                for value
                in bbox
            )
        )

    return tuple(
        sorted(
            boxes
        )
    )


def deduplicate_records(
    records: list[dict],
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:
    print_header(
        "EXACT DUPLICATE CHECK"
    )

    hash_groups: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for record in records:
        hash_groups[
            record["sha256"]
        ].append(
            record
        )

    duplicate_groups: list[
        dict
    ] = []

    annotation_conflicts: list[
        dict
    ] = []

    canonical_records: list[
        dict
    ] = []

    for image_hash, group in (
        hash_groups.items()
    ):
        ordered = sorted(
            group,
            key=lambda item:
                item[
                    "source_relative_path"
                ],
        )

        canonical = dict(
            ordered[0]
        )

        duplicates = [
            item[
                "source_relative_path"
            ]
            for item in ordered[1:]
        ]

        canonical[
            "duplicate_sources"
        ] = duplicates

        if len(ordered) > 1:
            signatures = {
                target_signature(
                    item
                )
                for item
                in ordered
            }

            duplicate_record = {
                "sha256":
                    image_hash,

                "canonical":
                    canonical[
                        "source_relative_path"
                    ],

                "duplicates":
                    duplicates,

                "copies":
                    len(
                        ordered
                    ),
            }

            duplicate_groups.append(
                duplicate_record
            )

            if len(signatures) > 1:
                annotation_conflicts.append(
                    {
                        **duplicate_record,
                        "reason":
                            "same_image_different_target_annotations",
                    }
                )

        canonical_records.append(
            canonical
        )

    removed_count = (
        len(records)
        - len(canonical_records)
    )

    print(
        f"Source images       : {len(records):,}"
    )

    print(
        f"Unique images       : {len(canonical_records):,}"
    )

    print(
        f"Duplicate copies    : {removed_count:,}"
    )

    print(
        f"Duplicate conflicts : "
        f"{len(annotation_conflicts):,}"
    )

    return (
        canonical_records,
        duplicate_groups,
        annotation_conflicts,
    )


# ============================================================
# LEAKAGE-SAFE BALANCED GROUP SPLIT
# ============================================================


def create_grouped_split(
    records: list[dict],
) -> tuple[
    dict[str, str],
    dict,
]:
    print_header(
        "LEAKAGE-SAFE 70 / 15 / 15 SPLIT"
    )

    groups: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for record in records:
        groups[
            record[
                "deployment"
            ]
        ].append(
            record
        )

    if len(groups) < 3:
        raise RuntimeError(
            "At least three independent deployment/video "
            "groups are required to create train/val/test."
        )

    group_stats = {}

    for group_name, group_records in (
        groups.items()
    ):
        group_stats[
            group_name
        ] = {
            "images":
                len(
                    group_records
                ),

            "target_boxes":
                sum(
                    item[
                        "target_count"
                    ]
                    for item
                    in group_records
                ),
        }

    total_images = len(
        records
    )

    total_boxes = sum(
        record[
            "target_count"
        ]
        for record
        in records
    )

    image_targets = {
        split:
            total_images
            * SPLIT_RATIOS[
                split
            ]
        for split
        in SPLIT_NAMES
    }

    box_targets = {
        split:
            total_boxes
            * SPLIT_RATIOS[
                split
            ]
        for split
        in SPLIT_NAMES
    }

    best_assignment = None
    best_score = float(
        "inf"
    )

    best_counts = None

    group_names = list(
        groups.keys()
    )

    for attempt in range(
        SPLIT_SEARCH_ATTEMPTS
    ):
        rng = random.Random(
            SPLIT_SEED
            + attempt
        )

        ordered_groups = (
            group_names.copy()
        )

        rng.shuffle(
            ordered_groups
        )

        # Larger groups first, with deterministic jitter so
        # repeated attempts can explore slightly different
        # assignments.
        jitter = {
            group:
                rng.random()
            for group
            in ordered_groups
        }

        ordered_groups.sort(
            key=lambda group: (
                max(
                    group_stats[
                        group
                    ][
                        "images"
                    ]
                    / max(
                        total_images,
                        1,
                    ),
                    group_stats[
                        group
                    ][
                        "target_boxes"
                    ]
                    / max(
                        total_boxes,
                        1,
                    ),
                ),
                jitter[
                    group
                ],
            ),
            reverse=True,
        )

        image_counts = {
            split: 0
            for split
            in SPLIT_NAMES
        }

        box_counts = {
            split: 0
            for split
            in SPLIT_NAMES
        }

        group_counts = {
            split: 0
            for split
            in SPLIT_NAMES
        }

        assignment = {}

        for group in ordered_groups:
            group_images = (
                group_stats[
                    group
                ][
                    "images"
                ]
            )

            group_boxes = (
                group_stats[
                    group
                ][
                    "target_boxes"
                ]
            )

            candidates = []

            for split in SPLIT_NAMES:
                image_fill = (
                    image_counts[
                        split
                    ]
                    + group_images
                ) / max(
                    image_targets[
                        split
                    ],
                    1.0,
                )

                if total_boxes > 0:
                    box_fill = (
                        box_counts[
                            split
                        ]
                        + group_boxes
                    ) / max(
                        box_targets[
                            split
                        ],
                        1.0,
                    )
                else:
                    box_fill = (
                        image_fill
                    )

                fill_score = (
                    0.75
                    * image_fill
                    + 0.25
                    * box_fill
                )

                candidates.append(
                    (
                        fill_score,
                        rng.random(),
                        split,
                    )
                )

            _, _, selected_split = min(
                candidates
            )

            assignment[
                group
            ] = selected_split

            image_counts[
                selected_split
            ] += group_images

            box_counts[
                selected_split
            ] += group_boxes

            group_counts[
                selected_split
            ] += 1

        if any(
            group_counts[
                split
            ] == 0
            for split
            in SPLIT_NAMES
        ):
            continue

        image_error = sum(
            abs(
                image_counts[
                    split
                ]
                / total_images
                - SPLIT_RATIOS[
                    split
                ]
            )
            for split
            in SPLIT_NAMES
        )

        if total_boxes > 0:
            box_error = sum(
                abs(
                    box_counts[
                        split
                    ]
                    / total_boxes
                    - SPLIT_RATIOS[
                        split
                    ]
                )
                for split
                in SPLIT_NAMES
            )
        else:
            box_error = 0.0

        score = (
            image_error
            + 0.50
            * box_error
        )

        if score < best_score:
            best_score = score

            best_assignment = (
                assignment.copy()
            )

            best_counts = {
                "images":
                    image_counts.copy(),
                "target_boxes":
                    box_counts.copy(),
                "groups":
                    group_counts.copy(),
            }

    if (
        best_assignment is None
        or best_counts is None
    ):
        raise RuntimeError(
            "Unable to create a valid grouped split."
        )

    split_stats = {}

    for split in SPLIT_NAMES:
        image_count = (
            best_counts[
                "images"
            ][split]
        )

        target_count = (
            best_counts[
                "target_boxes"
            ][split]
        )

        split_stats[
            split
        ] = {
            "images":
                image_count,

            "image_percentage":
                round(
                    image_count
                    / total_images
                    * 100.0,
                    2,
                ),

            "target_boxes":
                target_count,

            "target_box_percentage":
                round(
                    (
                        target_count
                        / total_boxes
                        * 100.0
                    )
                    if total_boxes
                    else 0.0,
                    2,
                ),

            "deployment_groups":
                best_counts[
                    "groups"
                ][split],
        }

    for split in SPLIT_NAMES:
        values = split_stats[
            split
        ]

        print(
            f"{split:>5}: "
            f"{values['images']:>6,} images "
            f"({values['image_percentage']:>6.2f}%) | "
            f"{values['target_boxes']:>6,} fish boxes"
        )

    return (
        best_assignment,
        {
            "seed":
                SPLIT_SEED,

            "search_attempts":
                SPLIT_SEARCH_ATTEMPTS,

            "ratios":
                SPLIT_RATIOS,

            "score":
                best_score,

            "stats":
                split_stats,
        },
    )


# ============================================================
# APPLY SPLIT TO RECORDS
# ============================================================


def assign_splits(
    records: list[dict],
    group_assignment: dict[
        str,
        str,
    ],
) -> list[dict]:
    prepared_records = []

    for source_record in records:
        record = dict(
            source_record
        )

        record[
            "prepared_split"
        ] = group_assignment[
            record[
                "deployment"
            ]
        ]

        prepared_records.append(
            record
        )

    return prepared_records


# ============================================================
# LEAKAGE VERIFICATION
# ============================================================


def verify_split_leakage(
    records: list[dict],
) -> dict:
    deployments = {
        split: set()
        for split
        in SPLIT_NAMES
    }

    hashes = {
        split: set()
        for split
        in SPLIT_NAMES
    }

    for record in records:
        split = record[
            "prepared_split"
        ]

        deployments[
            split
        ].add(
            record[
                "deployment"
            ]
        )

        hashes[
            split
        ].add(
            record[
                "sha256"
            ]
        )

    deployment_overlap = {
        "train_val":
            sorted(
                deployments[
                    "train"
                ]
                & deployments[
                    "val"
                ]
            ),

        "train_test":
            sorted(
                deployments[
                    "train"
                ]
                & deployments[
                    "test"
                ]
            ),

        "val_test":
            sorted(
                deployments[
                    "val"
                ]
                & deployments[
                    "test"
                ]
            ),
    }

    hash_overlap = {
        "train_val":
            sorted(
                hashes[
                    "train"
                ]
                & hashes[
                    "val"
                ]
            ),

        "train_test":
            sorted(
                hashes[
                    "train"
                ]
                & hashes[
                    "test"
                ]
            ),

        "val_test":
            sorted(
                hashes[
                    "val"
                ]
                & hashes[
                    "test"
                ]
            ),
    }

    deployment_leakage = any(
        deployment_overlap.values()
    )

    duplicate_leakage = any(
        hash_overlap.values()
    )

    return {
        "deployment_leakage":
            deployment_leakage,

        "duplicate_hash_leakage":
            duplicate_leakage,

        "deployment_overlap":
            deployment_overlap,

        "hash_overlap":
            hash_overlap,
    }


# ============================================================
# PREPARED DATASET CREATION
# ============================================================


def reset_prepared_dataset() -> None:
    if PREPARED_ROOT.exists():
        shutil.rmtree(
            PREPARED_ROOT
        )

    for split in SPLIT_NAMES:
        (
            PREPARED_ROOT
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            PREPARED_ROOT
            / split
            / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    PREPARED_MANIFESTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREPARED_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREPARED_REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_identifier(
    value: object,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value),
    )


def prepared_image_name(
    record: dict,
) -> str:
    original_name = Path(
        record[
            "file_name"
        ]
    ).name

    return (
        f"{record['source_split']}"
        f"__{safe_identifier(record['source_image_id'])}"
        f"__{original_name}"
    )


def coco_bbox_to_yolo(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    x, y, box_width, box_height = bbox

    center_x = (
        x
        + box_width
        / 2.0
    ) / width

    center_y = (
        y
        + box_height
        / 2.0
    ) / height

    normalized_width = (
        box_width
        / width
    )

    normalized_height = (
        box_height
        / height
    )

    return (
        min(
            max(
                center_x,
                0.0,
            ),
            1.0,
        ),

        min(
            max(
                center_y,
                0.0,
            ),
            1.0,
        ),

        min(
            max(
                normalized_width,
                0.0,
            ),
            1.0,
        ),

        min(
            max(
                normalized_height,
                0.0,
            ),
            1.0,
        ),
    )


def materialize_image(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    if USE_HARDLINKS:
        try:
            os.link(
                source,
                destination,
            )

            return "hardlink"

        except OSError:
            pass

    shutil.copy2(
        source,
        destination,
    )

    return "copy"


def materialize_prepared_dataset(
    records: list[dict],
) -> tuple[
    list[dict],
    dict,
]:
    print_header(
        "BUILDING PREPARED DATASET"
    )

    reset_prepared_dataset()

    enriched_records: list[
        dict
    ] = []

    materialization_counts = Counter()

    split_manifests: dict[
        str,
        list[str],
    ] = {
        split: []
        for split
        in SPLIT_NAMES
    }

    source_split_manifests: dict[
        str,
        list[str],
    ] = {
        split: []
        for split
        in SPLIT_NAMES
    }

    for record in tqdm(
        records,
        desc="Preparing dataset",
        unit="img",
        dynamic_ncols=True,
    ):
        split = record[
            "prepared_split"
        ]

        source_image = (
            source_path_from_record(
                record
            )
        )

        image_name = (
            prepared_image_name(
                record
            )
        )

        destination_image = (
            PREPARED_ROOT
            / split
            / "images"
            / image_name
        )

        label_name = (
            Path(
                image_name
            ).stem
            + ".txt"
        )

        destination_label = (
            PREPARED_ROOT
            / split
            / "labels"
            / label_name
        )

        mode = materialize_image(
            source_image,
            destination_image,
        )

        materialization_counts[
            mode
        ] += 1

        label_lines = []

        output_annotations = []

        for annotation in record[
            "target_boxes"
        ]:
            yolo_box = (
                coco_bbox_to_yolo(
                    annotation[
                        "bbox_xywh"
                    ],
                    record[
                        "width"
                    ],
                    record[
                        "height"
                    ],
                )
            )

            label_lines.append(
                (
                    f"{FINAL_CLASS_ID} "
                    f"{yolo_box[0]:.8f} "
                    f"{yolo_box[1]:.8f} "
                    f"{yolo_box[2]:.8f} "
                    f"{yolo_box[3]:.8f}"
                )
            )

            output_annotations.append(
                {
                    "class_id":
                        FINAL_CLASS_ID,

                    "class_name":
                        FINAL_CLASS_NAME,

                    "source_category":
                        annotation[
                            "source_category"
                        ],

                    "source_annotation_id":
                        annotation[
                            "source_annotation_id"
                        ],

                    "bbox_xywh":
                        annotation[
                            "bbox_xywh"
                        ],

                    "bbox_yolo":
                        [
                            float(value)
                            for value
                            in yolo_box
                        ],
                }
            )

        destination_label.write_text(
            (
                "\n".join(
                    label_lines
                )
                + (
                    "\n"
                    if label_lines
                    else ""
                )
            ),
            encoding="utf-8",
        )

        image_relative = (
            destination_image
            .relative_to(
                PREPARED_ROOT
            )
            .as_posix()
        )

        label_relative = (
            destination_label
            .relative_to(
                PREPARED_ROOT
            )
            .as_posix()
        )

        enriched = dict(
            record
        )

        enriched[
            "prepared_image"
        ] = image_relative

        enriched[
            "prepared_label"
        ] = label_relative

        enriched[
            "normalized_annotations"
        ] = output_annotations

        enriched_records.append(
            enriched
        )

        split_manifests[
            split
        ].append(
            image_relative
        )

        source_split_manifests[
            split
        ].append(
            record[
                "source_relative_path"
            ]
        )

    for split in SPLIT_NAMES:
        split_manifests[
            split
        ].sort()

        source_split_manifests[
            split
        ].sort()

        (
            PREPARED_MANIFESTS_ROOT
            / f"{split}.txt"
        ).write_text(
            "\n".join(
                split_manifests[
                    split
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        MASTER_SPLITS_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            MASTER_SPLITS_ROOT
            / f"{split}.txt"
        ).write_text(
            "\n".join(
                source_split_manifests[
                    split
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    return (
        enriched_records,
        dict(
            materialization_counts
        ),
    )


# ============================================================
# DATASET FINGERPRINT
# ============================================================


def calculate_dataset_fingerprint(
    records: list[dict],
) -> str:
    digest = hashlib.sha256()

    digest.update(
        PREPARATION_VERSION.encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            SPLIT_SEED
        ).encode(
            "utf-8"
        )
    )

    for split in SPLIT_NAMES:
        digest.update(
            (
                f"{split}:"
                f"{SPLIT_RATIOS[split]}"
            ).encode(
                "utf-8"
            )
        )

    for record in sorted(
        records,
        key=lambda item:
            item[
                "prepared_image"
            ],
    ):
        digest.update(
            record[
                "sha256"
            ].encode(
                "utf-8"
            )
        )

        digest.update(
            record[
                "prepared_split"
            ].encode(
                "utf-8"
            )
        )

        digest.update(
            record[
                "prepared_image"
            ].encode(
                "utf-8"
            )
        )

        digest.update(
            repr(
                target_signature(
                    record
                )
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


# ============================================================
# OUTPUT VERIFICATION
# ============================================================


def verify_yolo_label(
    path: Path,
    expected_boxes: int,
) -> list[str]:
    errors = []

    lines = [
        line.strip()
        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) != expected_boxes:
        errors.append(
            (
                f"expected_{expected_boxes}_boxes"
                f"_found_{len(lines)}"
            )
        )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"line_{line_number}_not_5_columns"
            )

            continue

        try:
            class_id = int(
                parts[0]
            )

            values = [
                float(value)
                for value
                in parts[1:]
            ]

        except ValueError:
            errors.append(
                f"line_{line_number}_invalid_number"
            )

            continue

        if class_id != FINAL_CLASS_ID:
            errors.append(
                f"line_{line_number}_wrong_class"
            )

        x, y, width, height = values

        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            errors.append(
                f"line_{line_number}_out_of_range"
            )

    return errors


def verify_prepared_dataset(
    records: list[dict],
) -> dict:
    print_header(
        "PREPARED DATASET VERIFICATION"
    )

    errors = []

    split_counts = Counter()

    negative_images = 0

    target_boxes = 0

    for record in tqdm(
        records,
        desc="Verifying prepared dataset",
        unit="img",
        dynamic_ncols=True,
    ):
        split = record[
            "prepared_split"
        ]

        split_counts[
            split
        ] += 1

        if record[
            "target_count"
        ] == 0:
            negative_images += 1

        target_boxes += (
            record[
                "target_count"
            ]
        )

        image_path = (
            PREPARED_ROOT
            / record[
                "prepared_image"
            ]
        )

        label_path = (
            PREPARED_ROOT
            / record[
                "prepared_label"
            ]
        )

        if not image_path.exists():
            errors.append(
                {
                    "image":
                        record[
                            "prepared_image"
                        ],
                    "reason":
                        "missing_prepared_image",
                }
            )

            continue

        if not label_path.exists():
            errors.append(
                {
                    "image":
                        record[
                            "prepared_image"
                        ],
                    "reason":
                        "missing_label",
                }
            )

            continue

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            errors.append(
                {
                    "image":
                        record[
                            "prepared_image"
                        ],
                    "reason":
                        "unreadable_prepared_image",
                }
            )

            continue

        height, width = (
            image.shape[:2]
        )

        if (
            width
            != record[
                "width"
            ]
            or height
            != record[
                "height"
            ]
        ):
            errors.append(
                {
                    "image":
                        record[
                            "prepared_image"
                        ],
                    "reason":
                        "prepared_dimensions_changed",
                }
            )

        label_errors = (
            verify_yolo_label(
                label_path,
                record[
                    "target_count"
                ],
            )
        )

        for label_error in (
            label_errors
        ):
            errors.append(
                {
                    "image":
                        record[
                            "prepared_image"
                        ],
                    "reason":
                        label_error,
                }
            )

    return {
        "errors":
            errors,

        "error_count":
            len(
                errors
            ),

        "images":
            len(
                records
            ),

        "negative_images":
            negative_images,

        "target_boxes":
            target_boxes,

        "split_counts":
            dict(
                split_counts
            ),
    }


# ============================================================
# METADATA OUTPUT
# ============================================================


def write_dataset_yaml() -> None:
    configuration = {
        "path":
            "preprocessing/prepared_dataset",

        "train":
            "train/images",

        "val":
            "val/images",

        "test":
            "test/images",

        "names": {
            FINAL_CLASS_ID:
                FINAL_CLASS_NAME,
        },
    }

    DATASET_YAML.write_text(
        yaml.safe_dump(
            configuration,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_metadata(
    source_records: list[dict],
    prepared_records: list[dict],
    source_audit: dict,
    duplicate_groups: list[dict],
    split_info: dict,
    leakage_info: dict,
    materialization_info: dict,
    verification: dict,
) -> dict:
    fingerprint = (
        calculate_dataset_fingerprint(
            prepared_records
        )
    )

    image_index = []

    annotation_index = {}

    for record in prepared_records:
        image_index.append(
            {
                "prepared_image":
                    record[
                        "prepared_image"
                    ],

                "prepared_label":
                    record[
                        "prepared_label"
                    ],

                "split":
                    record[
                        "prepared_split"
                    ],

                "source_relative_path":
                    record[
                        "source_relative_path"
                    ],

                "source_split":
                    record[
                        "source_split"
                    ],

                "source_image_id":
                    record[
                        "source_image_id"
                    ],

                "width":
                    record[
                        "width"
                    ],

                "height":
                    record[
                        "height"
                    ],

                "deployment":
                    record[
                        "deployment"
                    ],

                "sha256":
                    record[
                        "sha256"
                    ],

                "target_count":
                    record[
                        "target_count"
                    ],

                "fish_count":
                    record[
                        "fish_count"
                    ],

                "small_fish_count":
                    record[
                        "small_fish_count"
                    ],

                "non_target_count":
                    record[
                        "non_target_count"
                    ],

                "duplicate_sources":
                    record[
                        "duplicate_sources"
                    ],
            }
        )

        annotation_index[
            record[
                "prepared_image"
            ]
        ] = (
            record[
                "normalized_annotations"
            ]
        )

    split_counts = Counter(
        record[
            "prepared_split"
        ]
        for record
        in prepared_records
    )

    split_target_counts = Counter()

    split_negative_counts = Counter()

    for record in prepared_records:
        split = record[
            "prepared_split"
        ]

        split_target_counts[
            split
        ] += record[
            "target_count"
        ]

        if record[
            "target_count"
        ] == 0:
            split_negative_counts[
                split
            ] += 1

    dataset_info = {
        "preparation_version":
            PREPARATION_VERSION,

        "prepared_at_utc":
            utc_now(),

        "dataset_name":
            "Brackish",

        "source_url":
            SOURCE_URL,

        "source_root":
            str(
                RAW_ROOT
            ),

        "prepared_root":
            str(
                PREPARED_ROOT
            ),

        "final_classes": {
            str(
                FINAL_CLASS_ID
            ):
                FINAL_CLASS_NAME,
        },

        "source_target_mapping": {
            "fish":
                FINAL_CLASS_NAME,
            "small_fish":
                FINAL_CLASS_NAME,
        },

        "split_seed":
            SPLIT_SEED,

        "requested_split_ratios":
            SPLIT_RATIOS,

        "images":
            len(
                prepared_records
            ),

        "target_boxes":
            sum(
                record[
                    "target_count"
                ]
                for record
                in prepared_records
            ),

        "negative_images":
            sum(
                1
                for record
                in prepared_records
                if record[
                    "target_count"
                ] == 0
            ),

        "split_image_counts":
            dict(
                split_counts
            ),

        "split_target_box_counts":
            dict(
                split_target_counts
            ),

        "split_negative_image_counts":
            dict(
                split_negative_counts
            ),

        "duplicate_groups":
            len(
                duplicate_groups
            ),

        "duplicate_copies_removed":
            (
                len(
                    source_records
                )
                - len(
                    prepared_records
                )
            ),

        "materialization":
            materialization_info,

        "dataset_fingerprint_sha256":
            fingerprint,
    }

    write_json(
        PREPARED_METADATA_ROOT
        / "image_index.json",
        image_index,
    )

    write_json(
        PREPARED_METADATA_ROOT
        / "annotation_index.json",
        annotation_index,
    )

    write_json(
        PREPARED_METADATA_ROOT
        / "dataset_info.json",
        dataset_info,
    )

    # Master/source metadata
    source_image_index = [
        {
            key: value
            for key, value
            in record.items()
            if key
            not in {
                "target_boxes",
            }
        }
        for record
        in source_records
    ]

    source_annotation_index = {
        record[
            "source_relative_path"
        ]:
            record[
                "target_boxes"
            ]
        for record
        in source_records
    }

    write_json(
        MASTER_METADATA_ROOT
        / "source_image_index.json",
        source_image_index,
    )

    write_json(
        MASTER_METADATA_ROOT
        / "source_annotation_index.json",
        source_annotation_index,
    )

    write_json(
        MASTER_METADATA_ROOT
        / "dataset_provenance.json",
        {
            "dataset":
                "Brackish",

            "source_url":
                SOURCE_URL,

            "source_root":
                str(
                    RAW_ROOT
                ),

            "audited_at_utc":
                source_audit[
                    "prepared_at_utc"
                ],

            "preparation_version":
                PREPARATION_VERSION,

            "target_categories":
                sorted(
                    TARGET_CATEGORY_NAMES
                ),

            "final_class":
                FINAL_CLASS_NAME,

            "split_seed":
                SPLIT_SEED,

            "split_ratios":
                SPLIT_RATIOS,
        },
    )

    preparation_report = {
        "status":
            (
                "PASS"
                if (
                    verification[
                        "error_count"
                    ] == 0
                    and not leakage_info[
                        "deployment_leakage"
                    ]
                    and not leakage_info[
                        "duplicate_hash_leakage"
                    ]
                )
                else "FAIL"
            ),

        "prepared_at_utc":
            utc_now(),

        "source_audit":
            {
                "usable_records":
                    source_audit[
                        "usable_records"
                    ],

                "source_split_counts":
                    source_audit[
                        "source_split_counts"
                    ],

                "source_category_annotation_counts":
                    source_audit[
                        "source_category_annotation_counts"
                    ],

                "issue_counts": {
                    key:
                        len(value)
                    for key, value
                    in source_audit[
                        "issues"
                    ].items()
                },
            },

        "duplicates":
            {
                "groups":
                    len(
                        duplicate_groups
                    ),

                "copies_removed":
                    (
                        len(
                            source_records
                        )
                        - len(
                            prepared_records
                        )
                    ),

                "details":
                    duplicate_groups,
            },

        "split":
            split_info,

        "leakage":
            leakage_info,

        "verification":
            verification,

        "materialization":
            materialization_info,

        "dataset_fingerprint_sha256":
            fingerprint,
    }

    write_json(
        PREPARED_REPORTS_ROOT
        / "preparation_report.json",
        preparation_report,
    )

    return preparation_report


# ============================================================
# SOURCE AUDIT OUTPUT
# ============================================================


def write_source_audit(
    audit: dict,
    duplicate_groups: list[dict] | None = None,
    duplicate_conflicts: list[dict] | None = None,
) -> None:
    report = dict(
        audit
    )

    report[
        "duplicate_groups"
    ] = (
        duplicate_groups
        if duplicate_groups
        is not None
        else []
    )

    report[
        "duplicate_annotation_conflicts"
    ] = (
        duplicate_conflicts
        if duplicate_conflicts
        is not None
        else []
    )

    MASTER_REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        MASTER_REPORTS_ROOT
        / "source_dataset_audit.json",
        report,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_final_summary(
    prepared_records: list[dict],
    duplicate_groups: list[dict],
    split_info: dict,
    verification: dict,
    report: dict,
) -> None:
    print_header(
        "DATASET PREPARATION SUMMARY"
    )

    print(
        f"Status              : "
        f"{report['status']}"
    )

    print(
        f"Prepared images     : "
        f"{len(prepared_records):,}"
    )

    print(
        f"Fish boxes          : "
        f"{verification['target_boxes']:,}"
    )

    print(
        f"Negative images     : "
        f"{verification['negative_images']:,}"
    )

    print(
        f"Duplicate groups    : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Verification errors : "
        f"{verification['error_count']:,}"
    )

    print()

    print(
        "Final split:"
    )

    for split in SPLIT_NAMES:
        stats = (
            split_info[
                "stats"
            ][split]
        )

        print(
            f"  {split:<5} "
            f"{stats['images']:>6,} images "
            f"({stats['image_percentage']:>6.2f}%) | "
            f"{stats['target_boxes']:>6,} fish boxes"
        )

    print()

    print(
        "Prepared dataset:"
    )

    print(
        PREPARED_ROOT
    )

    print()

    print(
        "Reports:"
    )

    print(
        PREPARED_REPORTS_ROOT
        / "preparation_report.json"
    )

    print(
        MASTER_REPORTS_ROOT
        / "source_dataset_audit.json"
    )

    print()

    if report[
        "status"
    ] == "PASS":
        print(
            "STAGE 1 CERTIFIED."
        )

        print()

        print(
            "Next stage:"
        )

        print(
            "  preprocessing/smoke_test.py"
        )

    else:
        print(
            "STAGE 1 FAILED."
        )

        print(
            "Do not continue to smoke-test preprocessing "
            "until the reported issues are resolved."
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print()
    print("=" * 72)
    print(" FISH DETECTION - DATASET PREPARATION")
    print("=" * 72)

    print()
    print(
        "Pipeline:"
    )

    print(
        "RAW BRACKISH"
    )

    print(
        "     ↓"
    )

    print(
        "AUDIT + DEDUPLICATE + GROUP SPLIT"
    )

    print(
        "     ↓"
    )

    print(
        "YOLO LABEL CONVERSION"
    )

    print(
        "     ↓"
    )

    print(
        "PREPARED DATASET"
    )

    # --------------------------------------------------------
    # 1. Source dataset
    # --------------------------------------------------------

    ensure_source_dataset()

    # --------------------------------------------------------
    # 2. Source audit
    # --------------------------------------------------------

    (
        source_records,
        source_audit,
    ) = audit_source_dataset()

    source_fatal = (
        source_fatal_issue_count(
            source_audit
        )
    )

    write_source_audit(
        source_audit
    )

    print()
    print(
        f"Usable source images : "
        f"{len(source_records):,}"
    )

    print(
        f"Fatal audit issues   : "
        f"{source_fatal:,}"
    )

    if source_fatal > 0:
        print()
        print(
            "SOURCE DATASET AUDIT FAILED."
        )

        print(
            "See:"
        )

        print(
            MASTER_REPORTS_ROOT
            / "source_dataset_audit.json"
        )

        raise SystemExit(
            1
        )

    # --------------------------------------------------------
    # 3. Exact duplicate detection
    # --------------------------------------------------------

    (
        unique_records,
        duplicate_groups,
        duplicate_conflicts,
    ) = deduplicate_records(
        source_records
    )

    write_source_audit(
        source_audit,
        duplicate_groups,
        duplicate_conflicts,
    )

    if duplicate_conflicts:
        print()
        print(
            "DUPLICATE ANNOTATION CONFLICTS FOUND."
        )

        print(
            "Same image bytes have different target labels."
        )

        print(
            "Dataset preparation stopped."
        )

        raise SystemExit(
            1
        )

    # --------------------------------------------------------
    # 4. Leakage-safe group split
    # --------------------------------------------------------

    (
        group_assignment,
        split_info,
    ) = create_grouped_split(
        unique_records
    )

    records_with_splits = (
        assign_splits(
            unique_records,
            group_assignment,
        )
    )

    leakage_info = (
        verify_split_leakage(
            records_with_splits
        )
    )

    print()
    print(
        "Deployment leakage : "
        f"{leakage_info['deployment_leakage']}"
    )

    print(
        "Duplicate leakage  : "
        f"{leakage_info['duplicate_hash_leakage']}"
    )

    if (
        leakage_info[
            "deployment_leakage"
        ]
        or leakage_info[
            "duplicate_hash_leakage"
        ]
    ):
        raise RuntimeError(
            "Split leakage detected. "
            "Prepared dataset will not be created."
        )

    # --------------------------------------------------------
    # 5. Build prepared dataset
    # --------------------------------------------------------

    (
        prepared_records,
        materialization_info,
    ) = materialize_prepared_dataset(
        records_with_splits
    )

    # --------------------------------------------------------
    # 6. dataset.yaml
    # --------------------------------------------------------

    write_dataset_yaml()

    # --------------------------------------------------------
    # 7. Verify every prepared image/label
    # --------------------------------------------------------

    verification = (
        verify_prepared_dataset(
            prepared_records
        )
    )

    # --------------------------------------------------------
    # 8. Metadata / report / fingerprint
    # --------------------------------------------------------

    preparation_report = (
        write_metadata(
            source_records=source_records,
            prepared_records=prepared_records,
            source_audit=source_audit,
            duplicate_groups=duplicate_groups,
            split_info=split_info,
            leakage_info=leakage_info,
            materialization_info=materialization_info,
            verification=verification,
        )
    )

    # --------------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------------

    print_final_summary(
        prepared_records,
        duplicate_groups,
        split_info,
        verification,
        preparation_report,
    )

    if (
        preparation_report[
            "status"
        ]
        != "PASS"
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()