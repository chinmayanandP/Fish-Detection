from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


# ============================================================
# CANONICAL PROJECT STRUCTURE
# ============================================================

CANONICAL_DIRECTORIES = [

    # ========================================================
    # MASTER DATASET
    # ========================================================

    "master_dataset/downloads",
    "master_dataset/raw/brackish",
    "master_dataset/splits",
    "master_dataset/metadata",
    "master_dataset/reports",

    # ========================================================
    # PREPROCESSING - PREPARED DATASET
    # ========================================================

    "preprocessing/prepared_dataset/train/images",
    "preprocessing/prepared_dataset/train/labels",

    "preprocessing/prepared_dataset/val/images",
    "preprocessing/prepared_dataset/val/labels",

    "preprocessing/prepared_dataset/test/images",
    "preprocessing/prepared_dataset/test/labels",

    "preprocessing/prepared_dataset/manifests",
    "preprocessing/prepared_dataset/metadata",
    "preprocessing/prepared_dataset/reports",

    # ========================================================
    # PREPROCESSING - SMOKE TEST
    # ========================================================

    "preprocessing/smoke_test_results/candidates",
    "preprocessing/smoke_test_results/winners",
    "preprocessing/smoke_test_results/comparison_grids",
    "preprocessing/smoke_test_results/reports",

    # ========================================================
    # PREPROCESSING - FINAL PREPROCESSED DATASET
    # ========================================================

    "preprocessing/preprocessed_dataset/train/images",
    "preprocessing/preprocessed_dataset/train/labels",

    "preprocessing/preprocessed_dataset/val/images",
    "preprocessing/preprocessed_dataset/val/labels",

    "preprocessing/preprocessed_dataset/test/images",
    "preprocessing/preprocessed_dataset/test/labels",

    "preprocessing/preprocessed_dataset/manifests",
    "preprocessing/preprocessed_dataset/metadata",
    "preprocessing/preprocessed_dataset/reports",

    # ========================================================
    # TRAINING
    # ========================================================

    "training/configs",

    "training/results/models",
    "training/results/plots",
    "training/results/logs",
    "training/results/checkpoints",

    # ========================================================
    # VALIDATION
    # ========================================================

    "validation/results/metrics",
    "validation/results/plots",
    "validation/results/confusion_matrix",
    "validation/results/predictions",

    # ========================================================
    # TESTING
    # ========================================================

    "testing/results/metrics",
    "testing/results/plots",
    "testing/results/predictions",
    "testing/results/annotated_images",

    # ========================================================
    # EXTERNAL VIDEO
    # ========================================================

    "external_video/input",

    "external_video/results/videos",
    "external_video/results/frames",
    "external_video/results/detections",
    "external_video/results/metrics",

    # ========================================================
    # GENERAL
    # ========================================================

    "configs",
    "tests",
    ".vscode",
]


# ============================================================
# DIRECTORIES THAT SHOULD KEEP .gitkeep
# ============================================================

GITKEEP_DIRECTORIES = [

    # Master dataset local files
    "master_dataset/downloads",
    "master_dataset/raw/brackish",

    # Prepared dataset
    "preprocessing/prepared_dataset/train/images",
    "preprocessing/prepared_dataset/train/labels",

    "preprocessing/prepared_dataset/val/images",
    "preprocessing/prepared_dataset/val/labels",

    "preprocessing/prepared_dataset/test/images",
    "preprocessing/prepared_dataset/test/labels",

    # Smoke test
    "preprocessing/smoke_test_results/candidates",
    "preprocessing/smoke_test_results/winners",
    "preprocessing/smoke_test_results/comparison_grids",

    # Final preprocessed dataset
    "preprocessing/preprocessed_dataset/train/images",
    "preprocessing/preprocessed_dataset/train/labels",

    "preprocessing/preprocessed_dataset/val/images",
    "preprocessing/preprocessed_dataset/val/labels",

    "preprocessing/preprocessed_dataset/test/images",
    "preprocessing/preprocessed_dataset/test/labels",

    # Training results
    "training/results/models",
    "training/results/plots",
    "training/results/logs",
    "training/results/checkpoints",

    # Validation
    "validation/results/metrics",
    "validation/results/plots",
    "validation/results/confusion_matrix",
    "validation/results/predictions",

    # Testing
    "testing/results/metrics",
    "testing/results/plots",
    "testing/results/predictions",
    "testing/results/annotated_images",

    # External video
    "external_video/input",
    "external_video/results/videos",
    "external_video/results/frames",
    "external_video/results/detections",
    "external_video/results/metrics",
]


# ============================================================
# EXPECTED SOURCE FILES
# ============================================================

EXPECTED_SOURCE_FILES = [

    # Exactly three preprocessing Python files
    "preprocessing/prepare_dataset.py",
    "preprocessing/smoke_test.py",
    "preprocessing/master_preprocessing.py",

    # Other stages
    "training/train.py",
    "validation/validate.py",
    "testing/test_model.py",
    "external_video/test_video.py",
]


# ============================================================
# OBSOLETE STRUCTURE
# ============================================================

OBSOLETE_DIRECTORIES = [
    "data",
    "experiments",
    "reports",
    "runs",
]


# ============================================================
# HELPER
# ============================================================


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def run_command(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=capture_output,
    )


# ============================================================
# 1. PROJECT STRUCTURE
# ============================================================


def ensure_project_structure() -> None:
    print_header(
        "PROJECT STRUCTURE"
    )

    created = 0
    existing = 0

    for relative_path in CANONICAL_DIRECTORIES:

        path = (
            PROJECT_ROOT
            / relative_path
        )

        if path.exists():
            existing += 1

        else:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            created += 1

            print(
                f"[CREATED] {relative_path}"
            )

    print()
    print(
        f"Existing directories : {existing}"
    )

    print(
        f"Created directories  : {created}"
    )


# ============================================================
# 2. GITKEEP
# ============================================================


def ensure_gitkeep_files() -> None:

    for relative_path in GITKEEP_DIRECTORIES:

        directory = (
            PROJECT_ROOT
            / relative_path
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        gitkeep = (
            directory
            / ".gitkeep"
        )

        if not gitkeep.exists():
            gitkeep.touch()


# ============================================================
# 3. DATASET YAML FILES
# ============================================================


def ensure_dataset_yaml_files() -> None:
    """
    Create dataset.yaml files only if they do not already exist.

    prepare_dataset.py and master_preprocessing.py may later
    rewrite/update these files with final metadata.
    """

    datasets = [

        (
            PROJECT_ROOT
            / "preprocessing"
            / "prepared_dataset"
            / "dataset.yaml"
        ),

        (
            PROJECT_ROOT
            / "preprocessing"
            / "preprocessed_dataset"
            / "dataset.yaml"
        ),
    ]

    content = """path: .

train: train/images
val: val/images
test: test/images

names:
  0: fish
"""

    for path in datasets:

        if not path.exists():

            path.write_text(
                content,
                encoding="utf-8",
            )

            print(
                f"[CREATED] "
                f"{path.relative_to(PROJECT_ROOT)}"
            )


# ============================================================
# 4. LEGACY STRUCTURE CHECK
# ============================================================


def check_obsolete_directories() -> None:

    print_header(
        "LEGACY STRUCTURE CHECK"
    )

    found = []

    for name in OBSOLETE_DIRECTORIES:

        path = (
            PROJECT_ROOT
            / name
        )

        if path.exists():

            found.append(
                path
            )

    if not found:

        print(
            "[OK] No obsolete project directories found."
        )

        return

    print(
        "[WARNING] Old project directories still exist:"
    )

    print()

    for path in found:

        print(
            f"  - {path.name}/"
        )

    print()

    print(
        "They will NOT be deleted automatically."
    )


# ============================================================
# 5. SOURCE FILE VERIFICATION
# ============================================================


def verify_source_files() -> None:

    print_header(
        "SOURCE FILE CHECK"
    )

    for relative_path in EXPECTED_SOURCE_FILES:

        path = (
            PROJECT_ROOT
            / relative_path
        )

        if path.exists():

            print(
                f"[OK] {relative_path}"
            )

        else:

            print(
                f"[MISSING] {relative_path}"
            )

    # ========================================================
    # IMPORTANT:
    # preprocessing/ must contain exactly 3 Python files.
    # ========================================================

    preprocessing_dir = (
        PROJECT_ROOT
        / "preprocessing"
    )

    found_python_files = sorted(
        file.name
        for file
        in preprocessing_dir.glob(
            "*.py"
        )
    )

    expected_python_files = sorted(
        [
            "prepare_dataset.py",
            "smoke_test.py",
            "master_preprocessing.py",
        ]
    )

    print()
    print(
        "Preprocessing Python files:"
    )

    for filename in found_python_files:

        print(
            f"  - {filename}"
        )

    if (
        found_python_files
        == expected_python_files
    ):

        print()
        print(
            "[OK] preprocessing/ contains exactly "
            "the three canonical Python files."
        )

    else:

        print()
        print(
            "[WARNING] preprocessing/ Python files "
            "do not match the canonical structure."
        )


# ============================================================
# 6. VIRTUAL ENVIRONMENT
# ============================================================


def ensure_virtual_environment() -> bool:

    print_header(
        "PYTHON ENVIRONMENT"
    )

    if VENV_PYTHON.exists():

        print(
            "[OK] Existing .venv found."
        )

        print(
            f"Python: {VENV_PYTHON}"
        )

        print()

        print(
            "The virtual environment will be preserved."
        )

        return False

    print(
        "[INFO] No .venv found."
    )

    print(
        "[INFO] Creating virtual environment..."
    )

    run_command(
        [
            sys.executable,
            "-m",
            "venv",
            str(
                VENV_DIR
            ),
        ]
    )

    if not VENV_PYTHON.exists():

        raise RuntimeError(
            "Virtual environment creation failed."
        )

    print(
        "[OK] Virtual environment created."
    )

    return True


# ============================================================
# 7. LOAD REQUIREMENTS
# ============================================================


def load_requirements() -> list[
    tuple[str, str]
]:

    if not REQUIREMENTS_FILE.exists():

        raise FileNotFoundError(
            f"requirements.txt not found: "
            f"{REQUIREMENTS_FILE}"
        )

    requirements = []

    for raw_line in REQUIREMENTS_FILE.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if line.startswith("-"):
            continue

        if "://" in line:
            continue

        match = re.match(
            r"^([A-Za-z0-9_.-]+)",
            line,
        )

        if not match:
            continue

        distribution_name = (
            match.group(1)
        )

        requirements.append(
            (
                distribution_name,
                line,
            )
        )

    return requirements


# ============================================================
# 8. PACKAGE CHECK
# ============================================================


def package_is_installed(
    distribution_name: str,
) -> bool:

    result = run_command(
        [
            str(
                VENV_PYTHON
            ),
            "-m",
            "pip",
            "show",
            distribution_name,
        ],
        check=False,
        capture_output=True,
    )

    return (
        result.returncode
        == 0
    )


def find_missing_requirements() -> list[str]:

    print_header(
        "DEPENDENCY CHECK"
    )

    missing = []

    for (
        distribution_name,
        requirement_spec,
    ) in load_requirements():

        if package_is_installed(
            distribution_name
        ):

            print(
                f"[OK] {distribution_name}"
            )

        else:

            print(
                f"[MISSING] {distribution_name}"
            )

            missing.append(
                requirement_spec
            )

    return missing


# ============================================================
# 9. INSTALL ONLY MISSING REQUIREMENTS
# ============================================================


def install_missing_requirements(
    missing: list[str],
) -> None:

    if not missing:

        print()
        print(
            "[OK] All requirements are already installed."
        )

        print(
            "No packages were reinstalled."
        )

        return

    print()
    print(
        "Installing ONLY missing packages:"
    )

    for requirement in missing:

        print(
            f"  - {requirement}"
        )

    run_command(
        [
            str(
                VENV_PYTHON
            ),
            "-m",
            "pip",
            "install",
            *missing,
        ]
    )


# ============================================================
# 10. PIP CHECK
# ============================================================


def run_pip_check() -> None:

    print_header(
        "DEPENDENCY CONSISTENCY"
    )

    result = run_command(
        [
            str(
                VENV_PYTHON
            ),
            "-m",
            "pip",
            "check",
        ],
        check=False,
        capture_output=True,
    )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    if result.returncode == 0:

        print(
            "[OK] pip dependency check passed."
        )

        if output:

            print(
                output
            )

    else:

        print(
            "[WARNING] Dependency conflicts detected:"
        )

        print()

        print(
            output
        )


# ============================================================
# 11. IMPORT SMOKE TEST
# ============================================================


def run_import_test() -> None:

    print_header(
        "PYTHON IMPORT SMOKE TEST"
    )

    code = """
import cv2
import numpy
import requests
import tqdm
import pandas
import matplotlib
import sklearn
import albumentations
import yaml
import ultralytics

print("Core Python environment imports: OK")
"""

    result = run_command(
        [
            str(
                VENV_PYTHON
            ),
            "-c",
            code,
        ],
        check=False,
        capture_output=True,
    )

    if result.returncode == 0:

        print(
            result.stdout.strip()
        )

    else:

        print(
            "[ERROR] Import smoke test failed."
        )

        print()

        print(
            result.stderr.strip()
        )


# ============================================================
# 12. VS CODE SETTINGS
# ============================================================


def ensure_vscode_settings() -> None:

    print_header(
        "VS CODE"
    )

    vscode = (
        PROJECT_ROOT
        / ".vscode"
    )

    vscode.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings = (
        vscode
        / "settings.json"
    )

    if settings.exists():

        print(
            "[OK] Existing .vscode/settings.json preserved."
        )

        return

    settings.write_text(
        """{
    "python.defaultInterpreterPath": "${workspaceFolder}\\\\.venv\\\\Scripts\\\\python.exe",
    "python.terminal.activateEnvironment": true,

    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,

    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true
    },

    "search.exclude": {
        "**/.venv": true,
        "master_dataset/raw": true,
        "preprocessing/prepared_dataset": true,
        "preprocessing/preprocessed_dataset": true,
        "preprocessing/smoke_test_results/candidates": true
    }
}
""",
        encoding="utf-8",
    )

    print(
        "[CREATED] .vscode/settings.json"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_summary(
    missing_packages: int,
) -> None:

    print()
    print("=" * 72)
    print(" PROJECT SETUP COMPLETE")
    print("=" * 72)

    print()

    print(
        "Project structure        : VERIFIED"
    )

    print(
        "Virtual environment       : PRESERVED"
    )

    print(
        "Prepared dataset stage    : READY"
    )

    print(
        "Smoke-test stage          : READY"
    )

    print(
        "Master preprocessing      : READY"
    )

    if missing_packages == 0:

        print(
            "Requirements             : ALREADY INSTALLED"
        )

    else:

        print(
            "Requirements             : REPAIRED"
        )

    print(
        "Dataset preparation       : NOT RUN"
    )

    print(
        "Preprocessing             : NOT RUN"
    )

    print(
        "Training                  : NOT RUN"
    )

    print()

    print(
        "Canonical preprocessing pipeline:"
    )

    print()

    print(
        "master_dataset/raw/brackish"
    )

    print(
        "            ↓"
    )

    print(
        "prepare_dataset.py"
    )

    print(
        "            ↓"
    )

    print(
        "prepared_dataset"
    )

    print(
        "            ↓"
    )

    print(
        "smoke_test.py"
    )

    print(
        "            ↓"
    )

    print(
        "master_preprocessing.py"
    )

    print(
        "            ↓"
    )

    print(
        "preprocessed_dataset"
    )

    print(
        "            ↓"
    )

    print(
        "training"
    )

    print()

    print(
        "Next stage:"
    )

    print(
        "  preprocessing/prepare_dataset.py"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()
    print("=" * 72)
    print(" FISH DETECTION - PROJECT SETUP")
    print("=" * 72)

    print()

    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    ensure_project_structure()

    ensure_gitkeep_files()

    ensure_dataset_yaml_files()

    check_obsolete_directories()

    verify_source_files()

    ensure_virtual_environment()

    missing = (
        find_missing_requirements()
    )

    install_missing_requirements(
        missing
    )

    run_pip_check()

    run_import_test()

    ensure_vscode_settings()

    print_summary(
        len(
            missing
        )
    )


if __name__ == "__main__":
    main()