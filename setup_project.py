from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


# ============================================================
# PROJECT FOLDER STRUCTURE
# ============================================================

FOLDERS = [
    # --------------------------------------------------------
    # MAIN MACHINE LEARNING PIPELINE
    # --------------------------------------------------------
    "preprocessing",
    "training",
    "validation",
    "testing",

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    "data",
    "data/downloads",
    "data/raw",
    "data/raw/brackish",
    "data/processed",
    "data/processed/brackish",

    # Dataset split manifests
    "data/splits",

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    "configs",

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------
    "experiments",

    # --------------------------------------------------------
    # TRAINING OUTPUTS
    # --------------------------------------------------------
    "runs",

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------
    "reports",
    "reports/figures",

    # --------------------------------------------------------
    # AUTOMATED TESTS
    # --------------------------------------------------------
    "tests",

    # --------------------------------------------------------
    # VS CODE
    # --------------------------------------------------------
    ".vscode",
]


# ============================================================
# PYTHON PACKAGE INITIALIZATION FILES
# ============================================================

PYTHON_INIT_FILES = [
    "preprocessing/__init__.py",
    "training/__init__.py",
    "validation/__init__.py",
    "testing/__init__.py",
    "tests/__init__.py",
]


# ============================================================
# EMPTY FOLDERS THAT MUST APPEAR IN GITHUB
# ============================================================

GITKEEP_FOLDERS = [
    "configs",
    "data/downloads",
    "data/raw/brackish",
    "data/processed/brackish",
    "data/splits",
    "experiments",
    "reports/figures",
]


# ============================================================
# REQUIREMENTS.TXT
# ============================================================

REQUIREMENTS = """\
numpy
opencv-python
pillow
requests
tqdm
pandas
matplotlib
scikit-learn
albumentations
pyyaml
ultralytics
"""


# ============================================================
# .GITIGNORE
# ============================================================

GITIGNORE = """\
# ============================================================
# PYTHON VIRTUAL ENVIRONMENT
# ============================================================

.venv/
venv/
env/


# ============================================================
# PYTHON CACHE
# ============================================================

__pycache__/
*.pyc
*.pyo
*.pyd


# ============================================================
# DATASET DOWNLOADS
# ============================================================

data/downloads/*
!data/downloads/.gitkeep


# ============================================================
# RAW DATASET
# ============================================================

data/raw/brackish/*
!data/raw/brackish/.gitkeep


# ============================================================
# PROCESSED DATASET
# ============================================================

data/processed/brackish/*
!data/processed/brackish/.gitkeep


# ============================================================
# IMPORTANT:
# data/splits is intentionally NOT ignored.
#
# train.txt
# val.txt
# test.txt
#
# should be shared between collaborators so both computers
# train and evaluate using exactly the same split.
# ============================================================


# ============================================================
# YOLO / TRAINING OUTPUTS
# ============================================================

runs/

*.pt
*.pth
*.onnx
*.engine
*.cache


# ============================================================
# LOGS / TEMPORARY FILES
# ============================================================

*.log
*.tmp
*.temp


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

.env


# ============================================================
# OPERATING SYSTEM FILES
# ============================================================

.DS_Store
Thumbs.db
desktop.ini
"""


# ============================================================
# VS CODE SETTINGS
# ============================================================

VSCODE_SETTINGS = {
    # Project-specific Python interpreter
    "python.defaultInterpreterPath": (
        "${workspaceFolder}\\.venv\\Scripts\\python.exe"
    ),

    # Automatically activate Python environment
    # whenever a new VS Code terminal is opened.
    "python.terminal.activateEnvironment": True,

    # Look for .venv inside the current workspace.
    "python-envs.workspaceSearchPaths": [
        "./**/.venv"
    ],

    # Python linting / formatting
    "editor.formatOnSave": True,

    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },

    # Testing configuration
    "python.testing.pytestEnabled": True,
    "python.testing.unittestEnabled": False,

    # Hide Python cache files
    "files.exclude": {
        "**/__pycache__": True,
        "**/*.pyc": True
    },

    # Do not search huge local datasets
    "search.exclude": {
        "data/raw": True,
        "data/processed": True,
        "runs": True
    }
}


# ============================================================
# VS CODE EXTENSION RECOMMENDATIONS
# ============================================================

VSCODE_EXTENSIONS = {
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "charliermarsh.ruff",
    ]
}


# ============================================================
# VS CODE AUTOMATIC SETUP TASK
# ============================================================

VSCODE_TASKS = {
    "version": "2.0.0",

    "tasks": [
        {
            "label": "Setup Fish Detection Project",

            "type": "shell",

            "command": "python",

            "args": [
                "setup_project.py"
            ],

            "options": {
                "cwd": "${workspaceFolder}"
            },

            "presentation": {
                "reveal": "silent",
                "panel": "dedicated",
                "clear": False
            },

            "problemMatcher": [],

            "runOptions": {
                "runOn": "folderOpen"
            }
        }
    ]
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories() -> None:
    print("\n[1/8] Creating project folders...\n")

    for folder in FOLDERS:
        path = PROJECT_ROOT / folder

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"[OK] {folder}")


# ============================================================
# CREATE .GITKEEP FILES
# ============================================================

def create_gitkeep_files() -> None:
    print("\n[2/8] Creating Git placeholder files...\n")

    for folder in GITKEEP_FOLDERS:
        folder_path = PROJECT_ROOT / folder

        folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        gitkeep = folder_path / ".gitkeep"

        gitkeep.touch(
            exist_ok=True,
        )

        print(
            f"[OK] {gitkeep.relative_to(PROJECT_ROOT)}"
        )


# ============================================================
# CREATE __INIT__.PY FILES
# ============================================================

def create_python_packages() -> None:
    print("\n[3/8] Creating Python package files...\n")

    for filename in PYTHON_INIT_FILES:
        file_path = PROJECT_ROOT / filename

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.touch(
            exist_ok=True,
        )

        print(f"[OK] {filename}")


# ============================================================
# WRITE TEXT FILE
# ============================================================

def write_text_file(
    relative_path: str,
    content: str,
) -> None:
    file_path = PROJECT_ROOT / relative_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    print(f"[OK] {relative_path}")


# ============================================================
# WRITE JSON FILE
# ============================================================

def write_json_file(
    relative_path: str,
    data: dict,
) -> None:
    file_path = PROJECT_ROOT / relative_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        json.dumps(
            data,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] {relative_path}")


# ============================================================
# CREATE SHARED PROJECT FILES
# ============================================================

def create_project_files() -> None:
    print("\n[4/8] Creating shared project files...\n")

    write_text_file(
        "requirements.txt",
        REQUIREMENTS,
    )

    write_text_file(
        ".gitignore",
        GITIGNORE,
    )

    write_json_file(
        ".vscode/settings.json",
        VSCODE_SETTINGS,
    )

    write_json_file(
        ".vscode/extensions.json",
        VSCODE_EXTENSIONS,
    )

    write_json_file(
        ".vscode/tasks.json",
        VSCODE_TASKS,
    )


# ============================================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================================

def create_virtual_environment() -> bool:
    print("\n[5/8] Checking Python environment...\n")

    if VENV_PYTHON.exists():
        print(
            f"[OK] Virtual environment already exists:\n"
            f"{VENV_DIR}"
        )

        return False

    print("[CREATE] Creating .venv...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            str(VENV_DIR),
        ],
        check=True,
    )

    if not VENV_PYTHON.exists():
        raise RuntimeError(
            "Virtual environment creation failed."
        )

    print(
        f"\n[OK] Virtual environment created:\n"
        f"{VENV_DIR}"
    )

    return True


# ============================================================
# INSTALL REQUIREMENTS
# ============================================================

def install_requirements(
    environment_created: bool,
) -> None:
    print("\n[6/8] Checking Python dependencies...\n")

    stamp_file = (
        VENV_DIR
        / ".requirements_installed"
    )

    requirements_modified = (
        REQUIREMENTS_FILE.stat().st_mtime
    )

    requirements_need_installing = (
        environment_created
        or not stamp_file.exists()
        or (
            stamp_file.stat().st_mtime
            < requirements_modified
        )
    )

    if not requirements_need_installing:
        print(
            "[OK] Requirements are already installed."
        )

        return

    print("[INSTALL] Updating pip...\n")

    subprocess.run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ],
        check=True,
    )

    print("\n[INSTALL] Installing requirements...\n")

    subprocess.run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ],
        check=True,
    )

    stamp_file.touch()

    print(
        "\n[OK] Requirements installed successfully."
    )


# ============================================================
# CREATE STARTER FILES
# ============================================================

def create_starter_files() -> None:
    print("\n[7/8] Creating workflow starter files...\n")

    starter_files = {
        "preprocessing/preprocess.py": (
            '"""Dataset preprocessing pipeline."""\n'
        ),

        "training/train.py": (
            '"""YOLO model training pipeline."""\n'
        ),

        "validation/validate.py": (
            '"""Model validation pipeline."""\n'
        ),

        "testing/test_model.py": (
            '"""Final model testing pipeline."""\n'
        ),
    }

    for relative_path, content in starter_files.items():
        file_path = PROJECT_ROOT / relative_path

        if file_path.exists():
            print(
                f"[SKIP] {relative_path}"
            )

            continue

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        print(
            f"[OK] {relative_path}"
        )


# ============================================================
# PRINT PROJECT SUMMARY
# ============================================================

def print_summary() -> None:
    print("\n[8/8] Setup completed.")

    print("\n" + "=" * 65)
    print(" UNDERWATER FISH DETECTION PROJECT READY")
    print("=" * 65)

    print(
        "\nPROJECT PIPELINE\n"
    )

    print(
        "RAW DATASET\n"
        "     |\n"
        "     v\n"
        "PREPROCESSING\n"
        "     |\n"
        "     v\n"
        "TRAINING\n"
        "     |\n"
        "     v\n"
        "VALIDATION\n"
        "     |\n"
        "     v\n"
        "TESTING\n"
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nPython environment:"
    )

    print(
        VENV_DIR
    )

    print(
        "\nRaw dataset:"
    )

    print(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "brackish"
    )

    print(
        "\nProcessed dataset:"
    )

    print(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "brackish"
    )

    print(
        "\nVS Code is configured to use:"
    )

    print(
        ".venv\\Scripts\\python.exe"
    )

    print(
        "\nIMPORTANT:"
        "\nClose existing VS Code terminals and open a new terminal."
        "\nThe .venv environment should then activate automatically."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 65)
    print(" FISH DETECTION PROJECT SETUP")
    print("=" * 65)

    create_directories()

    create_gitkeep_files()

    create_python_packages()

    create_project_files()

    environment_created = (
        create_virtual_environment()
    )

    install_requirements(
        environment_created
    )

    create_starter_files()

    print_summary()


if __name__ == "__main__":
    main()