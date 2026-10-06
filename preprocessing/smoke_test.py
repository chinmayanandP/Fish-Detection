from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_ROOT = PROJECT_ROOT / "preprocessing"

PREPARED_ROOT = PREPROCESSING_ROOT / "prepared_dataset"

IMAGE_INDEX_FILE = PREPARED_ROOT / "metadata" / "image_index.json"
ANNOTATION_INDEX_FILE = PREPARED_ROOT / "metadata" / "annotation_index.json"
DATASET_INFO_FILE = PREPARED_ROOT / "metadata" / "dataset_info.json"

SMOKE_ROOT = PREPROCESSING_ROOT / "smoke_test_results"

CANDIDATES_ROOT = SMOKE_ROOT / "candidates"
WINNERS_ROOT = SMOKE_ROOT / "winners"
GRIDS_ROOT = SMOKE_ROOT / "comparison_grids"
REPORTS_ROOT = SMOKE_ROOT / "reports"


# ============================================================
# POLICY VERSION
# ============================================================

PREPROCESSING_POLICY_VERSION = "1.4.3-calibration"


# ============================================================
# RANDOM SEEDS
# ============================================================

SMOKE_TEST_SEED = 42
CALIBRATION_SEED = 314159


# ============================================================
# SAMPLE COUNTS
# ============================================================

CALIBRATION_POSITIVE_COUNT = 150
CALIBRATION_NEGATIVE_COUNT = 50

CALIBRATION_TOTAL_COUNT = (
    CALIBRATION_POSITIVE_COUNT
    + CALIBRATION_NEGATIVE_COUNT
)

SMOKE_POSITIVE_COUNT = 14
SMOKE_NEGATIVE_COUNT = 6

SMOKE_TOTAL_COUNT = (
    SMOKE_POSITIVE_COUNT
    + SMOKE_NEGATIVE_COUNT
)

POSITIVE_PROFILE_POOL = 400
NEGATIVE_PROFILE_POOL = 200


# ============================================================
# v1.4.3 DEPLOYMENT-GROUP CALIBRATION CONTROLS
# ============================================================

MAX_CALIBRATION_IMAGES_PER_DEPLOYMENT = 6

DESIRED_TINY_CALIBRATION_IMAGES = 60
DESIRED_SMALL_CALIBRATION_IMAGES = 45


# ============================================================
# ALL 8 PREPROCESSING MODES
# ============================================================

PREPROCESSING_MODES = (
    "raw_baseline",
    "white_balance_conservative",
    "clahe_very_mild",
    "adaptive_gamma_light",
    "adaptive_gamma_standard",
    "bilateral_denoise_mild",
    "mild_dehaze_low",
    "mild_dehaze_medium",
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

NORMAL_RISK_MODES = {
    "adaptive_gamma_light",
    "adaptive_gamma_standard",
    "mild_dehaze_low",
    "mild_dehaze_medium",
}

HIGHER_RISK_MODES = {
    "white_balance_conservative",
    "clahe_very_mild",
    "bilateral_denoise_mild",
}

HIGH_RISK_EXTRA_IMPROVEMENT = 0.003
HIGH_RISK_ADVANTAGE_OVER_SAFE = 0.002

BENEFIT_TIE_EPSILON = 0.0005


# ============================================================
# FORBIDDEN OPERATIONS
# ============================================================

FORBIDDEN_OPERATIONS = (
    "rotation",
    "large_rotation",
    "perspective_transform",
    "affine_transform",
    "crop",
    "resolution_reduction",
    "aspect_ratio_change",
    "synthetic_noise",
    "heavy_noise",
    "heavy_blur",
    "heavy_denoise",
    "aggressive_sharpening",
    "extreme_colour_shift",
    "extreme_hue_shift",
    "extreme_saturation",
)


# ============================================================
# GLOBAL SAFETY LIMITS
# ============================================================

MIN_DETAIL_RETENTION = 0.85
MIN_EDGE_RECALL = 0.72

MIN_SHARPNESS_RATIO = 0.65
MAX_SHARPNESS_RATIO = 3.00

MAX_NOISE_RATIO = 2.00
MAX_EDGE_DENSITY_RATIO = 2.50

MAX_ALLOWED_BLACK_CLIPPING = 0.10
MAX_ALLOWED_WHITE_CLIPPING = 0.10

MAX_EXTRA_CLIPPING = 0.05

MIN_BRIGHTNESS = 0.06
MAX_BRIGHTNESS = 0.94

MAX_MEAN_LAB_SHIFT = 55.0
MAX_SATURATION_SHIFT = 0.25


# ============================================================
# LABEL-FREE LOCAL DETAIL SAFETY
# ============================================================

LOCAL_GRID_SIZES = (
    4,
    8,
)

LOCAL_WORKING_MAX_DIMENSION = 640

LOCAL_MEANINGFUL_TILE_KEEP_RATIO = 0.60
LOCAL_MIN_MEANINGFUL_TILES = 8

MIN_LOCAL_DETAIL_P10 = 0.78
MIN_LOCAL_EDGE_RECALL_P10 = 0.72

MIN_LOCAL_SHARPNESS_P10 = 0.72
MAX_LOCAL_SHARPNESS_P90 = 2.40

MAX_LOCAL_NOISE_P90 = 1.80

MIN_LOCAL_CONTRAST_P10 = 0.70
MAX_LOCAL_CONTRAST_P90 = 2.20


# ============================================================
# STRICTER HIGH-RISK LIMITS
# ============================================================

HIGH_RISK_MIN_LOCAL_DETAIL_P10 = 0.84
HIGH_RISK_MIN_LOCAL_EDGE_RECALL_P10 = 0.78

HIGH_RISK_MIN_LOCAL_SHARPNESS_P10 = 0.80
HIGH_RISK_MAX_LOCAL_SHARPNESS_P90 = 2.00

HIGH_RISK_MAX_LOCAL_NOISE_P90 = 1.55

HIGH_RISK_MIN_LOCAL_CONTRAST_P10 = 0.78
HIGH_RISK_MAX_LOCAL_CONTRAST_P90 = 1.80


# ============================================================
# OFFLINE FISH ROI / TINY-FISH SAFETY
#
# NEVER used by runtime selector.
# ============================================================

MIN_ROI_DETAIL_RETENTION = 0.82
MIN_ROI_EDGE_RECALL = 0.76

MIN_ROI_SHARPNESS_RATIO = 0.80
MAX_ROI_SHARPNESS_RATIO = 2.50

MAX_ROI_NOISE_RATIO = 1.80
MAX_ROI_LAB_SHIFT = 45.0

TINY_BOX_AREA_RATIO = 0.0015
SMALL_BOX_AREA_RATIO = 0.0075

MIN_TINY_DETAIL_RETENTION = 0.88
MIN_TINY_EDGE_RECALL = 0.82

MIN_TINY_SHARPNESS_RATIO = 0.85
MAX_TINY_SHARPNESS_RATIO = 2.20

MAX_TINY_NOISE_RATIO = 1.60
MAX_TINY_LAB_SHIFT = 40.0

ROI_PADDING_RATIO = 0.20
MIN_PROTECTED_ROI_SIZE = 24


# ============================================================
# v1.4.3 CLASS-NEUTRAL THRESHOLD SEARCH
# ============================================================

THRESHOLD_GRID = (
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
    0.0080,
    0.0100,
    0.0120,
    0.0140,
    0.0160,
    0.0180,
    0.0200,
    0.0220,
    0.0250,
    0.0300,
)

DEFAULT_MINIMUM_IMPROVEMENT = 0.020


# ============================================================
# CALIBRATION MINIMUM COVERAGE
#
# Prevents "zero damage" from being satisfied trivially by
# processing only one or two calibration images.
#
# This is CLASS NEUTRAL.
# ============================================================

MIN_CALIBRATION_PROCESSED_COUNT = 10
MIN_CALIBRATION_PROCESSED_RATE = 0.05


# ============================================================
# BENEFIT SCORE
#
# Safety is handled separately by hard gates.
# ============================================================

BENEFIT_WEIGHTS = {
    "uciqe_proxy": 0.24,
    "uiqm_proxy": 0.18,
    "brightness_quality": 0.16,
    "contrast_quality": 0.14,
    "entropy_quality": 0.10,
    "color_balance": 0.10,
    "sharpness_quality": 0.08,
}


# ============================================================
# UTILITIES
# ============================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


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


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def safe_ratio(
    numerator: float,
    denominator: float,
    minimum_denominator: float = 1e-6,
) -> float:
    return float(
        numerator
        / max(
            denominator,
            minimum_denominator,
        )
    )


def percentile(
    values: list[float],
    q: float,
    default: float,
) -> float:
    if not values:
        return float(
            default
        )

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def weighted_mean(
    values: list[
        tuple[
            float,
            float,
        ]
    ],
) -> float | None:
    if not values:
        return None

    denominator = sum(
        weight
        for _, weight
        in values
    )

    if denominator <= 0:
        return None

    numerator = sum(
        value * weight
        for value, weight
        in values
    )

    return float(
        numerator
        / denominator
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================


def ensure_uint8_bgr(
    image: np.ndarray,
) -> np.ndarray:
    if image is None:
        raise ValueError(
            "Image is None."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected 3-channel BGR image."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected exactly 3 channels."
        )

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    return image


def geometry_preserved(
    raw: np.ndarray,
    candidate: np.ndarray,
) -> bool:
    return (
        raw.shape
        == candidate.shape
    )


# ============================================================
# PREPROCESSING ALGORITHMS
# ============================================================


def white_balance_conservative(
    image: np.ndarray,
) -> np.ndarray:
    image = ensure_uint8_bgr(
        image
    )

    working = image.astype(
        np.float32
    )

    channel_means = working.mean(
        axis=(0, 1)
    )

    global_mean = float(
        channel_means.mean()
    )

    gains = (
        global_mean
        / np.maximum(
            channel_means,
            1e-6,
        )
    )

    gains = np.clip(
        gains,
        0.85,
        1.15,
    )

    balanced = (
        working
        * gains.reshape(
            1,
            1,
            3,
        )
    )

    return np.clip(
        balanced,
        0,
        255,
    ).astype(
        np.uint8
    )


def clahe_very_mild(
    image: np.ndarray,
) -> np.ndarray:
    image = ensure_uint8_bgr(
        image
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    (
        l_channel,
        a_channel,
        b_channel,
    ) = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.25,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )


def calculate_adaptive_gamma(
    image: np.ndarray,
) -> float:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = (
        float(
            np.mean(
                gray
            )
        )
        / 255.0
    )

    brightness = clamp(
        brightness,
        0.01,
        0.99,
    )

    desired_brightness = 0.48

    gamma = (
        math.log(
            desired_brightness
        )
        / math.log(
            brightness
        )
    )

    return clamp(
        gamma,
        0.75,
        1.25,
    )


def adaptive_gamma_variant(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:
    image = ensure_uint8_bgr(
        image
    )

    target_gamma = (
        calculate_adaptive_gamma(
            image
        )
    )

    gamma = (
        1.0
        + strength
        * (
            target_gamma
            - 1.0
        )
    )

    gamma = clamp(
        gamma,
        0.75,
        1.25,
    )

    lookup_table = np.array(
        [
            (
                (index / 255.0)
                ** gamma
            )
            * 255.0
            for index
            in range(256)
        ],
        dtype=np.uint8,
    )

    return cv2.LUT(
        image,
        lookup_table,
    )


def bilateral_denoise_mild(
    image: np.ndarray,
) -> np.ndarray:
    image = ensure_uint8_bgr(
        image
    )

    return cv2.bilateralFilter(
        image,
        d=3,
        sigmaColor=20,
        sigmaSpace=20,
    )


def mild_dehaze_variant(
    image: np.ndarray,
    *,
    blend_strength: float,
    omega: float,
) -> np.ndarray:
    image = ensure_uint8_bgr(
        image
    )

    normalized = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    dark_channel = np.min(
        normalized,
        axis=2,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15),
    )

    dark_channel = cv2.erode(
        dark_channel,
        kernel,
    )

    flattened_dark = (
        dark_channel.reshape(
            -1
        )
    )

    flattened_image = (
        normalized.reshape(
            -1,
            3,
        )
    )

    atmospheric_count = max(
        1,
        int(
            flattened_dark.size
            * 0.001
        ),
    )

    brightest_indices = np.argpartition(
        flattened_dark,
        -atmospheric_count,
    )[
        -atmospheric_count:
    ]

    atmospheric_light = np.mean(
        flattened_image[
            brightest_indices
        ],
        axis=0,
    )

    atmospheric_light = np.maximum(
        atmospheric_light,
        0.10,
    )

    normalized_by_airlight = (
        normalized
        / atmospheric_light.reshape(
            1,
            1,
            3,
        )
    )

    transmission_dark = np.min(
        normalized_by_airlight,
        axis=2,
    )

    transmission_dark = cv2.erode(
        transmission_dark,
        kernel,
    )

    transmission = (
        1.0
        - omega
        * transmission_dark
    )

    transmission = cv2.GaussianBlur(
        transmission,
        (5, 5),
        0,
    )

    transmission = np.clip(
        transmission,
        0.60,
        1.0,
    )

    recovered = (
        (
            normalized
            - atmospheric_light.reshape(
                1,
                1,
                3,
            )
        )
        / transmission[
            ...,
            None
        ]
        + atmospheric_light.reshape(
            1,
            1,
            3,
        )
    )

    recovered = np.clip(
        recovered,
        0.0,
        1.0,
    )

    blended = (
        normalized
        * (
            1.0
            - blend_strength
        )
        + recovered
        * blend_strength
    )

    return np.clip(
        blended
        * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================


def generate_candidates(
    image: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:
    raw = ensure_uint8_bgr(
        image
    )

    candidates = {
        "raw_baseline":
            raw.copy(),

        "white_balance_conservative":
            white_balance_conservative(
                raw
            ),

        "clahe_very_mild":
            clahe_very_mild(
                raw
            ),

        "adaptive_gamma_light":
            adaptive_gamma_variant(
                raw,
                strength=0.55,
            ),

        "adaptive_gamma_standard":
            adaptive_gamma_variant(
                raw,
                strength=1.00,
            ),

        "bilateral_denoise_mild":
            bilateral_denoise_mild(
                raw
            ),

        "mild_dehaze_low":
            mild_dehaze_variant(
                raw,
                blend_strength=0.20,
                omega=0.40,
            ),

        "mild_dehaze_medium":
            mild_dehaze_variant(
                raw,
                blend_strength=0.35,
                omega=0.50,
            ),
    }

    for mode, candidate in (
        candidates.items()
    ):
        if not geometry_preserved(
            raw,
            candidate,
        ):
            raise RuntimeError(
                f"Geometry changed in mode: "
                f"{mode}"
            )

    return candidates


# ============================================================
# QUALITY METRICS
# ============================================================


def entropy_score(
    gray: np.ndarray,
) -> float:
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    ).reshape(
        -1
    )

    total = float(
        histogram.sum()
    )

    if total <= 0:
        return 0.0

    probabilities = (
        histogram
        / total
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def brightness_quality(
    brightness: float,
) -> float:
    return clamp(
        1.0
        - abs(
            brightness
            - 0.50
        )
        / 0.50,
        0.0,
        1.0,
    )


def color_balance_score(
    image: np.ndarray,
) -> float:
    means = image.astype(
        np.float32
    ).mean(
        axis=(0, 1)
    )

    average = float(
        means.mean()
    )

    if average <= 1e-6:
        return 0.0

    imbalance = (
        float(
            np.std(
                means
            )
        )
        / average
    )

    return clamp(
        1.0
        - imbalance,
        0.0,
        1.0,
    )


def colorfulness_score(
    image: np.ndarray,
) -> float:
    b, g, r = cv2.split(
        image.astype(
            np.float32
        )
    )

    rg = r - g

    yb = (
        0.5
        * (
            r + g
        )
        - b
    )

    standard_component = math.sqrt(
        float(
            np.std(
                rg
            )
        )
        ** 2
        + float(
            np.std(
                yb
            )
        )
        ** 2
    )

    mean_component = math.sqrt(
        float(
            np.mean(
                rg
            )
        )
        ** 2
        + float(
            np.mean(
                yb
            )
        )
        ** 2
    )

    value = (
        standard_component
        + 0.3
        * mean_component
    )

    return clamp(
        value / 100.0,
        0.0,
        1.0,
    )


def mean_saturation(
    image: np.ndarray,
) -> float:
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    return float(
        np.mean(
            hsv[
                ...,
                1
            ]
        )
        / 255.0
    )


def estimate_noise(
    gray: np.ndarray,
) -> float:
    gray_float = gray.astype(
        np.float32
    )

    smooth = cv2.GaussianBlur(
        gray_float,
        (3, 3),
        0,
    )

    residual = (
        gray_float
        - smooth
    )

    return float(
        np.std(
            residual
        )
    )


def edge_density(
    gray: np.ndarray,
) -> float:
    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    return float(
        np.mean(
            edges > 0
        )
    )


def uciqe_proxy(
    image: np.ndarray,
) -> float:
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    luminance = (
        lab[
            ...,
            0
        ]
        / 255.0
    )

    a_channel = (
        lab[
            ...,
            1
        ]
        - 128.0
    )

    b_channel = (
        lab[
            ...,
            2
        ]
        - 128.0
    )

    chroma = np.sqrt(
        a_channel ** 2
        + b_channel ** 2
    )

    chroma_std = clamp(
        float(
            np.std(
                chroma
            )
        )
        / 64.0,
        0.0,
        1.0,
    )

    luminance_contrast = float(
        np.percentile(
            luminance,
            99,
        )
        - np.percentile(
            luminance,
            1,
        )
    )

    saturation = (
        chroma
        / (
            np.sqrt(
                chroma ** 2
                + (
                    luminance
                    * 255.0
                )
                ** 2
            )
            + 1e-6
        )
    )

    saturation_mean = clamp(
        float(
            np.mean(
                saturation
            )
        ),
        0.0,
        1.0,
    )

    score = (
        0.4680
        * chroma_std
        + 0.2745
        * luminance_contrast
        + 0.2576
        * saturation_mean
    )

    return clamp(
        float(
            score
        ),
        0.0,
        1.0,
    )


def compute_quality_metrics(
    image: np.ndarray,
) -> dict[
    str,
    float,
]:
    image = ensure_uint8_bgr(
        image
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = (
        float(
            np.mean(
                gray
            )
        )
        / 255.0
    )

    contrast = float(
        np.std(
            gray
        )
    )

    contrast_quality = clamp(
        contrast / 64.0,
        0.0,
        1.0,
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    sharpness_quality = clamp(
        math.log1p(
            sharpness
        )
        / math.log1p(
            1500.0
        ),
        0.0,
        1.0,
    )

    entropy = entropy_score(
        gray
    )

    entropy_quality = clamp(
        entropy / 8.0,
        0.0,
        1.0,
    )

    black_clipping = float(
        np.mean(
            gray <= 5
        )
    )

    white_clipping = float(
        np.mean(
            gray >= 250
        )
    )

    color_balance = (
        color_balance_score(
            image
        )
    )

    colorfulness = (
        colorfulness_score(
            image
        )
    )

    saturation = (
        mean_saturation(
            image
        )
    )

    noise = estimate_noise(
        gray
    )

    density = edge_density(
        gray
    )

    uciqe = uciqe_proxy(
        image
    )

    uiqm_proxy = (
        0.35
        * colorfulness
        + 0.35
        * sharpness_quality
        + 0.30
        * contrast_quality
    )

    return {
        "brightness":
            brightness,

        "brightness_quality":
            brightness_quality(
                brightness
            ),

        "contrast":
            contrast,

        "contrast_quality":
            contrast_quality,

        "sharpness":
            sharpness,

        "sharpness_quality":
            sharpness_quality,

        "entropy":
            entropy,

        "entropy_quality":
            entropy_quality,

        "black_clipping":
            black_clipping,

        "white_clipping":
            white_clipping,

        "color_balance":
            color_balance,

        "colorfulness":
            colorfulness,

        "saturation":
            saturation,

        "noise_estimate":
            noise,

        "edge_density":
            density,

        "uciqe_proxy":
            uciqe,

        "uiqm_proxy":
            clamp(
                uiqm_proxy,
                0.0,
                1.0,
            ),
    }


# ============================================================
# STRUCTURAL METRICS
# ============================================================


def working_pair(
    raw: np.ndarray,
    candidate: np.ndarray,
    maximum_dimension: int = 640,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    raw_gray = cv2.cvtColor(
        raw,
        cv2.COLOR_BGR2GRAY,
    )

    candidate_gray = cv2.cvtColor(
        candidate,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = (
        raw_gray.shape
    )

    current_maximum = max(
        height,
        width,
    )

    if current_maximum <= maximum_dimension:
        return (
            raw_gray,
            candidate_gray,
        )

    scale = (
        maximum_dimension
        / current_maximum
    )

    target_width = max(
        1,
        int(
            width
            * scale
        ),
    )

    target_height = max(
        1,
        int(
            height
            * scale
        ),
    )

    raw_gray = cv2.resize(
        raw_gray,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    candidate_gray = cv2.resize(
        candidate_gray,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return (
        raw_gray,
        candidate_gray,
    )


def gradient_magnitude(
    gray: np.ndarray,
) -> np.ndarray:
    x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    return cv2.magnitude(
        x,
        y,
    )


def gradient_correlation(
    raw_gray: np.ndarray,
    candidate_gray: np.ndarray,
) -> float:
    raw_magnitude = (
        gradient_magnitude(
            raw_gray
        )
        .reshape(
            -1
        )
    )

    candidate_magnitude = (
        gradient_magnitude(
            candidate_gray
        )
        .reshape(
            -1
        )
    )

    raw_std = float(
        np.std(
            raw_magnitude
        )
    )

    candidate_std = float(
        np.std(
            candidate_magnitude
        )
    )

    if (
        raw_std < 1e-6
        or candidate_std < 1e-6
    ):
        return (
            1.0
            if np.allclose(
                raw_magnitude,
                candidate_magnitude,
                atol=1.0,
            )
            else 0.0
        )

    correlation = float(
        np.corrcoef(
            raw_magnitude,
            candidate_magnitude,
        )[0, 1]
    )

    if not math.isfinite(
        correlation
    ):
        return 0.0

    return clamp(
        correlation,
        0.0,
        1.0,
    )


def relative_gradient_edge_recall(
    raw_gray: np.ndarray,
    candidate_gray: np.ndarray,
) -> float:
    raw_gradient = (
        gradient_magnitude(
            raw_gray
        )
    )

    candidate_gradient = (
        gradient_magnitude(
            candidate_gray
        )
    )

    raw_positive = raw_gradient[
        raw_gradient > 0
    ]

    candidate_positive = (
        candidate_gradient[
            candidate_gradient > 0
        ]
    )

    if raw_positive.size == 0:
        return 1.0

    raw_threshold = float(
        np.percentile(
            raw_positive,
            75,
        )
    )

    if candidate_positive.size:
        candidate_threshold = float(
            np.percentile(
                candidate_positive,
                75,
            )
        )

    else:
        candidate_threshold = float(
            "inf"
        )

    raw_edges = (
        raw_gradient
        >= raw_threshold
    )

    candidate_edges = (
        candidate_gradient
        >= candidate_threshold
    )

    raw_edge_count = int(
        np.count_nonzero(
            raw_edges
        )
    )

    if raw_edge_count == 0:
        return 1.0

    candidate_uint8 = (
        candidate_edges.astype(
            np.uint8
        )
        * 255
    )

    candidate_dilated = (
        cv2.dilate(
            candidate_uint8,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
            iterations=1,
        )
        > 0
    )

    matched = int(
        np.count_nonzero(
            raw_edges
            & candidate_dilated
        )
    )

    return clamp(
        matched
        / raw_edge_count,
        0.0,
        1.0,
    )


def mean_lab_shift(
    raw: np.ndarray,
    candidate: np.ndarray,
) -> float:
    raw_source = raw
    candidate_source = candidate

    height, width = (
        raw.shape[:2]
    )

    maximum_dimension = max(
        height,
        width,
    )

    if maximum_dimension > 640:

        scale = (
            640.0
            / maximum_dimension
        )

        target_size = (
            max(
                1,
                int(
                    width
                    * scale
                ),
            ),
            max(
                1,
                int(
                    height
                    * scale
                ),
            ),
        )

        raw_source = cv2.resize(
            raw,
            target_size,
            interpolation=cv2.INTER_AREA,
        )

        candidate_source = cv2.resize(
            candidate,
            target_size,
            interpolation=cv2.INTER_AREA,
        )

    raw_lab = cv2.cvtColor(
        raw_source,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    candidate_lab = cv2.cvtColor(
        candidate_source,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    difference = (
        raw_lab
        - candidate_lab
    )

    distance = np.sqrt(
        np.sum(
            difference ** 2,
            axis=2,
        )
    )

    return float(
        np.mean(
            distance
        )
    )


def compute_structure_metrics(
    raw: np.ndarray,
    candidate: np.ndarray,
    raw_quality: dict,
    candidate_quality: dict,
) -> dict:

    if not geometry_preserved(
        raw,
        candidate,
    ):
        return {
            "geometry_preserved":
                False,

            "detail_retention":
                0.0,

            "edge_recall":
                0.0,

            "sharpness_ratio":
                0.0,

            "noise_ratio":
                float(
                    "inf"
                ),

            "edge_density_ratio":
                float(
                    "inf"
                ),

            "mean_lab_shift":
                float(
                    "inf"
                ),

            "saturation_shift":
                float(
                    "inf"
                ),
        }

    (
        raw_gray,
        candidate_gray,
    ) = working_pair(
        raw,
        candidate,
    )

    return {
        "geometry_preserved":
            True,

        "detail_retention":
            gradient_correlation(
                raw_gray,
                candidate_gray,
            ),

        "edge_recall":
            relative_gradient_edge_recall(
                raw_gray,
                candidate_gray,
            ),

        "sharpness_ratio":
            safe_ratio(
                candidate_quality[
                    "sharpness"
                ],
                max(
                    raw_quality[
                        "sharpness"
                    ],
                    1.0,
                ),
            ),

        "noise_ratio":
            safe_ratio(
                candidate_quality[
                    "noise_estimate"
                ],
                max(
                    raw_quality[
                        "noise_estimate"
                    ],
                    0.25,
                ),
            ),

        "edge_density_ratio":
            safe_ratio(
                candidate_quality[
                    "edge_density"
                ],
                max(
                    raw_quality[
                        "edge_density"
                    ],
                    1e-4,
                ),
            ),

        "mean_lab_shift":
            mean_lab_shift(
                raw,
                candidate,
            ),

        "saturation_shift":
            abs(
                candidate_quality[
                    "saturation"
                ]
                - raw_quality[
                    "saturation"
                ]
            ),
    }


# ============================================================
# LOCAL DETAIL ANALYSIS
# ============================================================


def local_tile_metric(
    raw_tile: np.ndarray,
    candidate_tile: np.ndarray,
) -> dict:

    raw_gradient = (
        gradient_magnitude(
            raw_tile
        )
    )

    detail_energy = float(
        np.mean(
            raw_gradient
        )
    )

    raw_sharpness = float(
        cv2.Laplacian(
            raw_tile,
            cv2.CV_64F,
        ).var()
    )

    candidate_sharpness = float(
        cv2.Laplacian(
            candidate_tile,
            cv2.CV_64F,
        ).var()
    )

    raw_noise = estimate_noise(
        raw_tile
    )

    candidate_noise = estimate_noise(
        candidate_tile
    )

    raw_contrast = float(
        np.std(
            raw_tile
        )
    )

    candidate_contrast = float(
        np.std(
            candidate_tile
        )
    )

    return {
        "detail_energy":
            detail_energy,

        "detail_retention":
            gradient_correlation(
                raw_tile,
                candidate_tile,
            ),

        "edge_recall":
            relative_gradient_edge_recall(
                raw_tile,
                candidate_tile,
            ),

        "sharpness_ratio":
            safe_ratio(
                candidate_sharpness,
                max(
                    raw_sharpness,
                    1.0,
                ),
            ),

        "noise_ratio":
            safe_ratio(
                candidate_noise,
                max(
                    raw_noise,
                    0.25,
                ),
            ),

        "contrast_ratio":
            safe_ratio(
                candidate_contrast,
                max(
                    raw_contrast,
                    1.0,
                ),
            ),
    }


def collect_local_tiles(
    raw_gray: np.ndarray,
    candidate_gray: np.ndarray,
) -> list[dict]:

    height, width = (
        raw_gray.shape
    )

    records = []

    for grid_size in LOCAL_GRID_SIZES:

        for row in range(
            grid_size
        ):

            y1 = int(
                round(
                    row
                    * height
                    / grid_size
                )
            )

            y2 = int(
                round(
                    (
                        row + 1
                    )
                    * height
                    / grid_size
                )
            )

            for column in range(
                grid_size
            ):

                x1 = int(
                    round(
                        column
                        * width
                        / grid_size
                    )
                )

                x2 = int(
                    round(
                        (
                            column + 1
                        )
                        * width
                        / grid_size
                    )
                )

                raw_tile = (
                    raw_gray[
                        y1:y2,
                        x1:x2,
                    ]
                )

                candidate_tile = (
                    candidate_gray[
                        y1:y2,
                        x1:x2,
                    ]
                )

                if (
                    raw_tile.shape[0] < 8
                    or raw_tile.shape[1] < 8
                ):
                    continue

                metrics = (
                    local_tile_metric(
                        raw_tile,
                        candidate_tile,
                    )
                )

                metrics[
                    "grid_size"
                ] = grid_size

                records.append(
                    metrics
                )

    return records


def compute_local_detail_metrics(
    raw: np.ndarray,
    candidate: np.ndarray,
) -> dict:

    (
        raw_gray,
        candidate_gray,
    ) = working_pair(
        raw,
        candidate,
        LOCAL_WORKING_MAX_DIMENSION,
    )

    tiles = collect_local_tiles(
        raw_gray,
        candidate_gray,
    )

    if not tiles:
        return {
            "meaningful_tile_count":
                0,

            "detail_retention_p10":
                1.0,

            "detail_retention_median":
                1.0,

            "edge_recall_p10":
                1.0,

            "edge_recall_median":
                1.0,

            "sharpness_ratio_p10":
                1.0,

            "sharpness_ratio_median":
                1.0,

            "sharpness_ratio_p90":
                1.0,

            "noise_ratio_median":
                1.0,

            "noise_ratio_p90":
                1.0,

            "contrast_ratio_p10":
                1.0,

            "contrast_ratio_median":
                1.0,

            "contrast_ratio_p90":
                1.0,
        }

    ordered = sorted(
        tiles,
        key=lambda item:
            item[
                "detail_energy"
            ],
        reverse=True,
    )

    keep_count = max(
        LOCAL_MIN_MEANINGFUL_TILES,
        int(
            math.ceil(
                len(
                    ordered
                )
                * LOCAL_MEANINGFUL_TILE_KEEP_RATIO
            )
        ),
    )

    keep_count = min(
        keep_count,
        len(
            ordered
        ),
    )

    meaningful = ordered[
        :keep_count
    ]

    details = [
        item[
            "detail_retention"
        ]
        for item
        in meaningful
    ]

    edges = [
        item[
            "edge_recall"
        ]
        for item
        in meaningful
    ]

    sharpness = [
        item[
            "sharpness_ratio"
        ]
        for item
        in meaningful
    ]

    noise = [
        item[
            "noise_ratio"
        ]
        for item
        in meaningful
    ]

    contrast = [
        item[
            "contrast_ratio"
        ]
        for item
        in meaningful
    ]

    return {
        "meaningful_tile_count":
            len(
                meaningful
            ),

        "detail_retention_p10":
            percentile(
                details,
                10,
                1.0,
            ),

        "detail_retention_median":
            percentile(
                details,
                50,
                1.0,
            ),

        "edge_recall_p10":
            percentile(
                edges,
                10,
                1.0,
            ),

        "edge_recall_median":
            percentile(
                edges,
                50,
                1.0,
            ),

        "sharpness_ratio_p10":
            percentile(
                sharpness,
                10,
                1.0,
            ),

        "sharpness_ratio_median":
            percentile(
                sharpness,
                50,
                1.0,
            ),

        "sharpness_ratio_p90":
            percentile(
                sharpness,
                90,
                1.0,
            ),

        "noise_ratio_median":
            percentile(
                noise,
                50,
                1.0,
            ),

        "noise_ratio_p90":
            percentile(
                noise,
                90,
                1.0,
            ),

        "contrast_ratio_p10":
            percentile(
                contrast,
                10,
                1.0,
            ),

        "contrast_ratio_median":
            percentile(
                contrast,
                50,
                1.0,
            ),

        "contrast_ratio_p90":
            percentile(
                contrast,
                90,
                1.0,
            ),
    }


# ============================================================
# LABEL-FREE RUNTIME SAFETY GATE
# ============================================================


def runtime_damage_gate(
    mode: str,
    raw_quality: dict,
    candidate_quality: dict,
    structure: dict,
    local_detail: dict,
) -> tuple[
    bool,
    list[str],
]:

    reasons = []

    if not structure[
        "geometry_preserved"
    ]:
        reasons.append(
            "geometry_changed"
        )

    if (
        structure[
            "detail_retention"
        ]
        < MIN_DETAIL_RETENTION
    ):
        reasons.append(
            "global_detail_loss"
        )

    if (
        structure[
            "edge_recall"
        ]
        < MIN_EDGE_RECALL
    ):
        reasons.append(
            "global_edge_loss"
        )

    if (
        structure[
            "sharpness_ratio"
        ]
        < MIN_SHARPNESS_RATIO
    ):
        reasons.append(
            "global_blur"
        )

    if (
        structure[
            "sharpness_ratio"
        ]
        > MAX_SHARPNESS_RATIO
    ):
        reasons.append(
            "global_over_sharpening"
        )

    if (
        structure[
            "noise_ratio"
        ]
        > MAX_NOISE_RATIO
    ):
        reasons.append(
            "global_noise_increase"
        )

    if (
        structure[
            "edge_density_ratio"
        ]
        > MAX_EDGE_DENSITY_RATIO
    ):
        reasons.append(
            "edge_explosion_or_artifacts"
        )

    candidate_black = (
        candidate_quality[
            "black_clipping"
        ]
    )

    candidate_white = (
        candidate_quality[
            "white_clipping"
        ]
    )

    if (
        candidate_black
        > MAX_ALLOWED_BLACK_CLIPPING
    ):
        reasons.append(
            "excessive_black_clipping"
        )

    if (
        candidate_white
        > MAX_ALLOWED_WHITE_CLIPPING
    ):
        reasons.append(
            "excessive_white_clipping"
        )

    extra_black = max(
        0.0,
        candidate_black
        - raw_quality[
            "black_clipping"
        ],
    )

    extra_white = max(
        0.0,
        candidate_white
        - raw_quality[
            "white_clipping"
        ],
    )

    if (
        extra_black
        + extra_white
        > MAX_EXTRA_CLIPPING
    ):
        reasons.append(
            "added_clipping_above_limit"
        )

    if (
        candidate_quality[
            "brightness"
        ]
        < MIN_BRIGHTNESS
    ):
        reasons.append(
            "image_too_dark"
        )

    if (
        candidate_quality[
            "brightness"
        ]
        > MAX_BRIGHTNESS
    ):
        reasons.append(
            "image_too_bright"
        )

    if (
        structure[
            "mean_lab_shift"
        ]
        > MAX_MEAN_LAB_SHIFT
    ):
        reasons.append(
            "extreme_colour_shift"
        )

    if (
        structure[
            "saturation_shift"
        ]
        > MAX_SATURATION_SHIFT
    ):
        reasons.append(
            "extreme_saturation_shift"
        )

    if mode in HIGHER_RISK_MODES:

        min_local_detail = (
            HIGH_RISK_MIN_LOCAL_DETAIL_P10
        )

        min_local_edge = (
            HIGH_RISK_MIN_LOCAL_EDGE_RECALL_P10
        )

        min_local_sharpness = (
            HIGH_RISK_MIN_LOCAL_SHARPNESS_P10
        )

        max_local_sharpness = (
            HIGH_RISK_MAX_LOCAL_SHARPNESS_P90
        )

        max_local_noise = (
            HIGH_RISK_MAX_LOCAL_NOISE_P90
        )

        min_local_contrast = (
            HIGH_RISK_MIN_LOCAL_CONTRAST_P10
        )

        max_local_contrast = (
            HIGH_RISK_MAX_LOCAL_CONTRAST_P90
        )

    else:

        min_local_detail = (
            MIN_LOCAL_DETAIL_P10
        )

        min_local_edge = (
            MIN_LOCAL_EDGE_RECALL_P10
        )

        min_local_sharpness = (
            MIN_LOCAL_SHARPNESS_P10
        )

        max_local_sharpness = (
            MAX_LOCAL_SHARPNESS_P90
        )

        max_local_noise = (
            MAX_LOCAL_NOISE_P90
        )

        min_local_contrast = (
            MIN_LOCAL_CONTRAST_P10
        )

        max_local_contrast = (
            MAX_LOCAL_CONTRAST_P90
        )

    if (
        local_detail[
            "detail_retention_p10"
        ]
        < min_local_detail
    ):
        reasons.append(
            "local_detail_loss"
        )

    if (
        local_detail[
            "edge_recall_p10"
        ]
        < min_local_edge
    ):
        reasons.append(
            "local_edge_loss"
        )

    if (
        local_detail[
            "sharpness_ratio_p10"
        ]
        < min_local_sharpness
    ):
        reasons.append(
            "local_blur"
        )

    if (
        local_detail[
            "sharpness_ratio_p90"
        ]
        > max_local_sharpness
    ):
        reasons.append(
            "local_over_sharpening"
        )

    if (
        local_detail[
            "noise_ratio_p90"
        ]
        > max_local_noise
    ):
        reasons.append(
            "local_noise_increase"
        )

    if (
        local_detail[
            "contrast_ratio_p10"
        ]
        < min_local_contrast
    ):
        reasons.append(
            "local_contrast_loss"
        )

    if (
        local_detail[
            "contrast_ratio_p90"
        ]
        > max_local_contrast
    ):
        reasons.append(
            "local_contrast_explosion"
        )

    return (
        len(
            reasons
        )
        == 0,
        reasons,
    )


# ============================================================
# BENEFIT SCORE
#
# Safety is NOT penalised here.
# Safety has its own hard gate.
# ============================================================


def compute_benefit_score(
    raw_quality: dict,
    candidate_quality: dict,
) -> tuple[
    float,
    dict,
]:

    components = {}

    for metric, weight in (
        BENEFIT_WEIGHTS.items()
    ):

        delta = (
            candidate_quality[
                metric
            ]
            - raw_quality[
                metric
            ]
        )

        weighted_delta = (
            weight
            * delta
        )

        components[
            metric
        ] = {
            "raw":
                float(
                    raw_quality[
                        metric
                    ]
                ),

            "candidate":
                float(
                    candidate_quality[
                        metric
                    ]
                ),

            "delta":
                float(
                    delta
                ),

            "weight":
                float(
                    weight
                ),

            "weighted_delta":
                float(
                    weighted_delta
                ),
        }

    score = sum(
        component[
            "weighted_delta"
        ]
        for component
        in components.values()
    )

    return (
        float(
            score
        ),
        components,
    )


# ============================================================
# SAFETY MARGIN
#
# Tie-breaking / reporting only.
# ============================================================


def ratio_closeness_to_one(
    ratio: float,
    max_log_distance: float = math.log(
        2.5
    ),
) -> float:

    ratio = max(
        ratio,
        1e-6,
    )

    distance = abs(
        math.log(
            ratio
        )
    )

    return clamp(
        1.0
        - distance
        / max_log_distance,
        0.0,
        1.0,
    )


def safety_margin_score(
    mode: str,
    structure: dict,
    local_detail: dict,
) -> float:

    if mode in HIGHER_RISK_MODES:

        min_local_detail = (
            HIGH_RISK_MIN_LOCAL_DETAIL_P10
        )

        min_local_edge = (
            HIGH_RISK_MIN_LOCAL_EDGE_RECALL_P10
        )

        max_local_noise = (
            HIGH_RISK_MAX_LOCAL_NOISE_P90
        )

    else:

        min_local_detail = (
            MIN_LOCAL_DETAIL_P10
        )

        min_local_edge = (
            MIN_LOCAL_EDGE_RECALL_P10
        )

        max_local_noise = (
            MAX_LOCAL_NOISE_P90
        )

    global_detail_margin = clamp(
        (
            structure[
                "detail_retention"
            ]
            - MIN_DETAIL_RETENTION
        )
        / max(
            1.0
            - MIN_DETAIL_RETENTION,
            1e-6,
        ),
        0.0,
        1.0,
    )

    global_edge_margin = clamp(
        (
            structure[
                "edge_recall"
            ]
            - MIN_EDGE_RECALL
        )
        / max(
            1.0
            - MIN_EDGE_RECALL,
            1e-6,
        ),
        0.0,
        1.0,
    )

    local_detail_margin = clamp(
        (
            local_detail[
                "detail_retention_p10"
            ]
            - min_local_detail
        )
        / max(
            1.0
            - min_local_detail,
            1e-6,
        ),
        0.0,
        1.0,
    )

    local_edge_margin = clamp(
        (
            local_detail[
                "edge_recall_p10"
            ]
            - min_local_edge
        )
        / max(
            1.0
            - min_local_edge,
            1e-6,
        ),
        0.0,
        1.0,
    )

    local_sharpness_closeness = (
        ratio_closeness_to_one(
            local_detail[
                "sharpness_ratio_median"
            ]
        )
    )

    local_noise_ratio = (
        local_detail[
            "noise_ratio_p90"
        ]
    )

    if local_noise_ratio <= 1.0:

        noise_margin = 1.0

    else:

        noise_margin = clamp(
            1.0
            - (
                local_noise_ratio
                - 1.0
            )
            / max(
                max_local_noise
                - 1.0,
                1e-6,
            ),
            0.0,
            1.0,
        )

    margins = (
        global_detail_margin,
        global_edge_margin,
        local_detail_margin,
        local_edge_margin,
        local_sharpness_closeness,
        noise_margin,
    )

    return float(
        np.mean(
            margins
        )
    )


# ============================================================
# LABEL-FREE CANDIDATE EVALUATION
# ============================================================


def evaluate_candidate(
    mode: str,
    raw: np.ndarray,
    candidate: np.ndarray,
    raw_quality: dict,
) -> dict:

    candidate_quality = (
        compute_quality_metrics(
            candidate
        )
    )

    structure = (
        compute_structure_metrics(
            raw,
            candidate,
            raw_quality,
            candidate_quality,
        )
    )

    local_detail = (
        compute_local_detail_metrics(
            raw,
            candidate,
        )
    )

    (
        gate_passed,
        rejection_reasons,
    ) = runtime_damage_gate(
        mode,
        raw_quality,
        candidate_quality,
        structure,
        local_detail,
    )

    (
        benefit_score,
        benefit_components,
    ) = compute_benefit_score(
        raw_quality,
        candidate_quality,
    )

    margin_score = (
        safety_margin_score(
            mode,
            structure,
            local_detail,
        )
        if gate_passed
        else 0.0
    )

    return {
        "risk_class":
            (
                "higher"
                if mode
                in HIGHER_RISK_MODES
                else "normal"
            ),

        "quality":
            candidate_quality,

        "structure":
            structure,

        "local_detail":
            local_detail,

        "damage_gate_passed":
            gate_passed,

        "rejection_reasons":
            rejection_reasons,

        "benefit_score":
            float(
                benefit_score
            ),

        "improvement_score":
            float(
                benefit_score
            ),

        "benefit_components":
            benefit_components,

        "safety_margin_score":
            float(
                margin_score
            ),
    }


def evaluate_all_candidates(
    raw_image: np.ndarray,
) -> tuple[
    dict[
        str,
        np.ndarray,
    ],
    dict[
        str,
        dict,
    ],
]:

    raw_image = ensure_uint8_bgr(
        raw_image
    )

    candidates = generate_candidates(
        raw_image
    )

    raw_quality = (
        compute_quality_metrics(
            raw_image
        )
    )

    evaluations = {
        "raw_baseline": {
            "risk_class":
                "baseline",

            "quality":
                raw_quality,

            "structure": {
                "geometry_preserved":
                    True,

                "detail_retention":
                    1.0,

                "edge_recall":
                    1.0,

                "sharpness_ratio":
                    1.0,

                "noise_ratio":
                    1.0,

                "edge_density_ratio":
                    1.0,

                "mean_lab_shift":
                    0.0,

                "saturation_shift":
                    0.0,
            },

            "local_detail": {
                "meaningful_tile_count":
                    0,

                "detail_retention_p10":
                    1.0,

                "detail_retention_median":
                    1.0,

                "edge_recall_p10":
                    1.0,

                "edge_recall_median":
                    1.0,

                "sharpness_ratio_p10":
                    1.0,

                "sharpness_ratio_median":
                    1.0,

                "sharpness_ratio_p90":
                    1.0,

                "noise_ratio_median":
                    1.0,

                "noise_ratio_p90":
                    1.0,

                "contrast_ratio_p10":
                    1.0,

                "contrast_ratio_median":
                    1.0,

                "contrast_ratio_p90":
                    1.0,
            },

            "damage_gate_passed":
                True,

            "rejection_reasons":
                [],

            "benefit_score":
                0.0,

            "improvement_score":
                0.0,

            "benefit_components":
                {},

            "safety_margin_score":
                1.0,
        }
    }

    for mode in PREPROCESSING_MODES:

        if mode == "raw_baseline":
            continue

        evaluations[
            mode
        ] = evaluate_candidate(
            mode,
            raw_image,
            candidates[
                mode
            ],
            raw_quality,
        )

    return (
        candidates,
        evaluations,
    )


# ============================================================
# RISK-AWARE WINNER SELECTION
# ============================================================


def choose_best_by_benefit_and_margin(
    candidates: list[
        tuple[
            str,
            dict,
        ]
    ],
) -> tuple[
    str,
    float,
] | None:

    if not candidates:
        return None

    best_benefit = max(
        float(
            evaluation[
                "benefit_score"
            ]
        )
        for _, evaluation
        in candidates
    )

    near_ties = [
        (
            mode,
            evaluation,
        )
        for mode, evaluation
        in candidates
        if (
            best_benefit
            - float(
                evaluation[
                    "benefit_score"
                ]
            )
            <= BENEFIT_TIE_EPSILON
        )
    ]

    mode_priority = {
        mode: index
        for index, mode
        in enumerate(
            PREPROCESSING_MODES
        )
    }

    (
        winner_mode,
        winner_evaluation,
    ) = max(
        near_ties,
        key=lambda item: (
            float(
                item[1][
                    "safety_margin_score"
                ]
            ),
            -mode_priority[
                item[0]
            ],
        ),
    )

    return (
        winner_mode,
        float(
            winner_evaluation[
                "benefit_score"
            ]
        ),
    )


def choose_winner_from_evaluations(
    evaluations: dict,
    minimum_improvement: float,
) -> str:

    normal_candidates = []

    for mode in PREPROCESSING_MODES:

        if mode not in NORMAL_RISK_MODES:
            continue

        evaluation = (
            evaluations[
                mode
            ]
        )

        if not evaluation[
            "damage_gate_passed"
        ]:
            continue

        if (
            float(
                evaluation[
                    "benefit_score"
                ]
            )
            < minimum_improvement
        ):
            continue

        normal_candidates.append(
            (
                mode,
                evaluation,
            )
        )

    safe_result = (
        choose_best_by_benefit_and_margin(
            normal_candidates
        )
    )

    if safe_result is None:

        best_safe_mode = (
            "raw_baseline"
        )

        best_safe_score = 0.0

    else:

        (
            best_safe_mode,
            best_safe_score,
        ) = safe_result

    risky_threshold = (
        minimum_improvement
        + HIGH_RISK_EXTRA_IMPROVEMENT
    )

    risky_candidates = []

    for mode in PREPROCESSING_MODES:

        if mode not in HIGHER_RISK_MODES:
            continue

        evaluation = (
            evaluations[
                mode
            ]
        )

        if not evaluation[
            "damage_gate_passed"
        ]:
            continue

        score = float(
            evaluation[
                "benefit_score"
            ]
        )

        if score < risky_threshold:
            continue

        if (
            best_safe_mode
            != "raw_baseline"
            and score
            < (
                best_safe_score
                + HIGH_RISK_ADVANTAGE_OVER_SAFE
            )
        ):
            continue

        risky_candidates.append(
            (
                mode,
                evaluation,
            )
        )

    risky_result = (
        choose_best_by_benefit_and_margin(
            risky_candidates
        )
    )

    if risky_result is None:
        return best_safe_mode

    (
        best_risky_mode,
        best_risky_score,
    ) = risky_result

    if best_safe_mode == "raw_baseline":
        return best_risky_mode

    if (
        best_risky_score
        >= (
            best_safe_score
            + HIGH_RISK_ADVANTAGE_OVER_SAFE
        )
    ):
        return best_risky_mode

    return best_safe_mode


def select_best_candidate(
    raw_image: np.ndarray,
    minimum_improvement: float,
) -> tuple[
    str,
    np.ndarray,
    dict,
    dict,
]:
    """
    AUTHORITATIVE LABEL-FREE RUNTIME SELECTOR.

    No annotation argument is accepted.

    Intended for:
      - master_preprocessing.py
      - external_video/test_video.py
    """

    (
        candidates,
        evaluations,
    ) = evaluate_all_candidates(
        raw_image
    )

    winner_mode = (
        choose_winner_from_evaluations(
            evaluations,
            minimum_improvement,
        )
    )

    return (
        winner_mode,
        candidates[
            winner_mode
        ],
        evaluations,
        candidates,
    )


# ============================================================
# OFFLINE ROI / TINY-FISH AUDIT
# ============================================================


def bbox_area_ratio(
    bbox: list,
    image_width: int,
    image_height: int,
) -> float:

    _, _, width, height = map(
        float,
        bbox,
    )

    image_area = max(
        float(
            image_width
            * image_height
        ),
        1.0,
    )

    return (
        width
        * height
        / image_area
    )


def classify_bbox_size(
    bbox: list,
    image_width: int,
    image_height: int,
) -> str:

    ratio = bbox_area_ratio(
        bbox,
        image_width,
        image_height,
    )

    if ratio < TINY_BOX_AREA_RATIO:
        return "tiny"

    if ratio < SMALL_BOX_AREA_RATIO:
        return "small"

    return "medium_large"


def bbox_size_counts(
    annotations: list,
    image_width: int,
    image_height: int,
) -> dict:

    counts = Counter()

    for annotation in annotations:

        bbox = annotation.get(
            "bbox_xywh"
        )

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(
                bbox
            )
            != 4
        ):
            continue

        size_class = (
            classify_bbox_size(
                bbox,
                image_width,
                image_height,
            )
        )

        counts[
            size_class
        ] += 1

    return {
        "tiny":
            counts[
                "tiny"
            ],

        "small":
            counts[
                "small"
            ],

        "medium_large":
            counts[
                "medium_large"
            ],
    }


def protected_roi_bounds(
    bbox: list,
    image_width: int,
    image_height: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    x, y, width, height = map(
        float,
        bbox,
    )

    center_x = (
        x
        + width / 2.0
    )

    center_y = (
        y
        + height / 2.0
    )

    protected_width = max(
        width
        * (
            1.0
            + 2.0
            * ROI_PADDING_RATIO
        ),
        float(
            MIN_PROTECTED_ROI_SIZE
        ),
    )

    protected_height = max(
        height
        * (
            1.0
            + 2.0
            * ROI_PADDING_RATIO
        ),
        float(
            MIN_PROTECTED_ROI_SIZE
        ),
    )

    x1 = int(
        math.floor(
            center_x
            - protected_width
            / 2.0
        )
    )

    y1 = int(
        math.floor(
            center_y
            - protected_height
            / 2.0
        )
    )

    x2 = int(
        math.ceil(
            center_x
            + protected_width
            / 2.0
        )
    )

    y2 = int(
        math.ceil(
            center_y
            + protected_height
            / 2.0
        )
    )

    return (
        max(
            0,
            x1,
        ),
        max(
            0,
            y1,
        ),
        min(
            image_width,
            x2,
        ),
        min(
            image_height,
            y2,
        ),
    )


def compute_one_roi_metrics(
    raw_roi: np.ndarray,
    candidate_roi: np.ndarray,
) -> dict:

    raw_gray = cv2.cvtColor(
        raw_roi,
        cv2.COLOR_BGR2GRAY,
    )

    candidate_gray = cv2.cvtColor(
        candidate_roi,
        cv2.COLOR_BGR2GRAY,
    )

    raw_sharpness = float(
        cv2.Laplacian(
            raw_gray,
            cv2.CV_64F,
        ).var()
    )

    candidate_sharpness = float(
        cv2.Laplacian(
            candidate_gray,
            cv2.CV_64F,
        ).var()
    )

    raw_noise = estimate_noise(
        raw_gray
    )

    candidate_noise = estimate_noise(
        candidate_gray
    )

    raw_contrast = float(
        np.std(
            raw_gray
        )
    )

    candidate_contrast = float(
        np.std(
            candidate_gray
        )
    )

    return {
        "detail_retention":
            gradient_correlation(
                raw_gray,
                candidate_gray,
            ),

        "edge_recall":
            relative_gradient_edge_recall(
                raw_gray,
                candidate_gray,
            ),

        "sharpness_ratio":
            safe_ratio(
                candidate_sharpness,
                max(
                    raw_sharpness,
                    1.0,
                ),
            ),

        "noise_ratio":
            safe_ratio(
                candidate_noise,
                max(
                    raw_noise,
                    0.25,
                ),
            ),

        "contrast_ratio":
            safe_ratio(
                candidate_contrast,
                max(
                    raw_contrast,
                    1.0,
                ),
            ),

        "lab_shift":
            mean_lab_shift(
                raw_roi,
                candidate_roi,
            ),
    }


def compute_roi_metrics(
    raw: np.ndarray,
    candidate: np.ndarray,
    annotations: list,
) -> dict:

    if not annotations:
        return {
            "has_targets":
                False,

            "target_count":
                0,

            "tiny_count":
                0,

            "small_count":
                0,

            "medium_large_count":
                0,
        }

    image_height, image_width = (
        raw.shape[:2]
    )

    keys = (
        "detail_retention",
        "edge_recall",
        "sharpness_ratio",
        "noise_ratio",
        "contrast_ratio",
        "lab_shift",
    )

    all_metrics = {
        key: []
        for key
        in keys
    }

    tiny_metrics = {
        key: []
        for key
        in keys
    }

    size_counts = Counter()

    valid_targets = 0

    for annotation in annotations:

        bbox = annotation.get(
            "bbox_xywh"
        )

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(
                bbox
            )
            != 4
        ):
            continue

        size_class = (
            classify_bbox_size(
                bbox,
                image_width,
                image_height,
            )
        )

        size_counts[
            size_class
        ] += 1

        if size_class == "tiny":
            weight = 2.0

        elif size_class == "small":
            weight = 1.5

        else:
            weight = 1.0

        (
            x1,
            y1,
            x2,
            y2,
        ) = protected_roi_bounds(
            bbox,
            image_width,
            image_height,
        )

        if (
            x2 - x1 < 3
            or y2 - y1 < 3
        ):
            continue

        raw_roi = raw[
            y1:y2,
            x1:x2,
        ]

        candidate_roi = (
            candidate[
                y1:y2,
                x1:x2,
            ]
        )

        if (
            raw_roi.size == 0
            or candidate_roi.size == 0
        ):
            continue

        metrics = (
            compute_one_roi_metrics(
                raw_roi,
                candidate_roi,
            )
        )

        valid_targets += 1

        for key in keys:

            all_metrics[
                key
            ].append(
                (
                    float(
                        metrics[
                            key
                        ]
                    ),
                    weight,
                )
            )

            if size_class == "tiny":

                tiny_metrics[
                    key
                ].append(
                    (
                        float(
                            metrics[
                                key
                            ]
                        ),
                        1.0,
                    )
                )

    output = {
        "has_targets":
            valid_targets > 0,

        "target_count":
            valid_targets,

        "tiny_count":
            size_counts[
                "tiny"
            ],

        "small_count":
            size_counts[
                "small"
            ],

        "medium_large_count":
            size_counts[
                "medium_large"
            ],
    }

    for key in keys:

        output[
            key
        ] = weighted_mean(
            all_metrics[
                key
            ]
        )

        output[
            f"tiny_{key}"
        ] = weighted_mean(
            tiny_metrics[
                key
            ]
        )

    return output


def offline_roi_audit(
    raw: np.ndarray,
    candidate: np.ndarray,
    annotations: list,
) -> dict:

    roi = compute_roi_metrics(
        raw,
        candidate,
        annotations,
    )

    reasons = []

    if not roi.get(
        "has_targets",
        False,
    ):
        return {
            "applicable":
                False,

            "passed":
                True,

            "reasons":
                [],

            "metrics":
                roi,
        }

    if (
        roi.get(
            "detail_retention"
        )
        is not None
        and roi[
            "detail_retention"
        ]
        < MIN_ROI_DETAIL_RETENTION
    ):
        reasons.append(
            "fish_roi_detail_loss"
        )

    if (
        roi.get(
            "edge_recall"
        )
        is not None
        and roi[
            "edge_recall"
        ]
        < MIN_ROI_EDGE_RECALL
    ):
        reasons.append(
            "fish_roi_edge_loss"
        )

    if (
        roi.get(
            "sharpness_ratio"
        )
        is not None
        and roi[
            "sharpness_ratio"
        ]
        < MIN_ROI_SHARPNESS_RATIO
    ):
        reasons.append(
            "fish_roi_blur"
        )

    if (
        roi.get(
            "sharpness_ratio"
        )
        is not None
        and roi[
            "sharpness_ratio"
        ]
        > MAX_ROI_SHARPNESS_RATIO
    ):
        reasons.append(
            "fish_roi_over_sharpening"
        )

    if (
        roi.get(
            "noise_ratio"
        )
        is not None
        and roi[
            "noise_ratio"
        ]
        > MAX_ROI_NOISE_RATIO
    ):
        reasons.append(
            "fish_roi_noise_increase"
        )

    if (
        roi.get(
            "lab_shift"
        )
        is not None
        and roi[
            "lab_shift"
        ]
        > MAX_ROI_LAB_SHIFT
    ):
        reasons.append(
            "fish_roi_colour_shift"
        )

    if roi.get(
        "tiny_count",
        0,
    ) > 0:

        if (
            roi.get(
                "tiny_detail_retention"
            )
            is not None
            and roi[
                "tiny_detail_retention"
            ]
            < MIN_TINY_DETAIL_RETENTION
        ):
            reasons.append(
                "tiny_fish_detail_loss"
            )

        if (
            roi.get(
                "tiny_edge_recall"
            )
            is not None
            and roi[
                "tiny_edge_recall"
            ]
            < MIN_TINY_EDGE_RECALL
        ):
            reasons.append(
                "tiny_fish_edge_loss"
            )

        if (
            roi.get(
                "tiny_sharpness_ratio"
            )
            is not None
            and roi[
                "tiny_sharpness_ratio"
            ]
            < MIN_TINY_SHARPNESS_RATIO
        ):
            reasons.append(
                "tiny_fish_blur"
            )

        if (
            roi.get(
                "tiny_sharpness_ratio"
            )
            is not None
            and roi[
                "tiny_sharpness_ratio"
            ]
            > MAX_TINY_SHARPNESS_RATIO
        ):
            reasons.append(
                "tiny_fish_over_sharpening"
            )

        if (
            roi.get(
                "tiny_noise_ratio"
            )
            is not None
            and roi[
                "tiny_noise_ratio"
            ]
            > MAX_TINY_NOISE_RATIO
        ):
            reasons.append(
                "tiny_fish_noise_increase"
            )

        if (
            roi.get(
                "tiny_lab_shift"
            )
            is not None
            and roi[
                "tiny_lab_shift"
            ]
            > MAX_TINY_LAB_SHIFT
        ):
            reasons.append(
                "tiny_fish_colour_shift"
            )

    return {
        "applicable":
            True,

        "passed":
            len(
                reasons
            )
            == 0,

        "reasons":
            reasons,

        "metrics":
            roi,
    }


# ============================================================
# METADATA LOADERS
# ============================================================


def load_image_index() -> list:

    if not IMAGE_INDEX_FILE.exists():
        raise FileNotFoundError(
            "Prepared image index is missing. "
            "Run prepare_dataset.py first."
        )

    return json.loads(
        IMAGE_INDEX_FILE.read_text(
            encoding="utf-8"
        )
    )


def load_annotation_index() -> dict:

    if not ANNOTATION_INDEX_FILE.exists():
        raise FileNotFoundError(
            "Prepared annotation index is missing. "
            "Run prepare_dataset.py first."
        )

    return json.loads(
        ANNOTATION_INDEX_FILE.read_text(
            encoding="utf-8"
        )
    )


def load_dataset_info() -> dict:

    if not DATASET_INFO_FILE.exists():
        return {}

    return json.loads(
        DATASET_INFO_FILE.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# REPRESENTATIVE SMOKE PROFILING
# ============================================================


def sample_profile(
    image: np.ndarray,
) -> dict:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    means = image.astype(
        np.float32
    ).mean(
        axis=(0, 1)
    )

    average = max(
        float(
            means.mean()
        ),
        1e-6,
    )

    return {
        "brightness":
            float(
                np.mean(
                    gray
                )
            )
            / 255.0,

        "contrast":
            float(
                np.std(
                    gray
                )
            ),

        "sharpness":
            float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()
            ),

        "colour_cast":
            float(
                np.std(
                    means
                )
            )
            / average,
    }


def profile_record(
    record: dict,
    annotation_index: dict,
) -> dict:

    image_path = (
        PREPARED_ROOT
        / record[
            "prepared_image"
        ]
    )

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read: "
            f"{image_path}"
        )

    annotations = (
        annotation_index.get(
            record[
                "prepared_image"
            ],
            [],
        )
    )

    sizes = bbox_size_counts(
        annotations,
        record[
            "width"
        ],
        record[
            "height"
        ],
    )

    output = dict(
        record
    )

    output[
        "_profile"
    ] = sample_profile(
        image
    )

    output[
        "_tiny_count"
    ] = sizes[
        "tiny"
    ]

    output[
        "_small_count"
    ] = sizes[
        "small"
    ]

    return output


def choose_ranked(
    records: list,
    selected_paths: set,
    key_function,
    reason: str,
    reverse: bool = False,
) -> dict | None:

    ordered = sorted(
        records,
        key=key_function,
        reverse=reverse,
    )

    for record in ordered:

        path = record[
            "prepared_image"
        ]

        if path in selected_paths:
            continue

        selected_paths.add(
            path
        )

        output = dict(
            record
        )

        output[
            "smoke_test_reason"
        ] = reason

        return output

    return None


def fill_random_samples(
    source_records: list,
    selected: list,
    selected_paths: set,
    desired_count: int,
    rng: random.Random,
    reason: str,
) -> None:

    available = [
        record
        for record
        in source_records
        if record[
            "prepared_image"
        ]
        not in selected_paths
    ]

    rng.shuffle(
        available
    )

    for record in available:

        if (
            len(
                selected
            )
            >= desired_count
        ):
            break

        selected_paths.add(
            record[
                "prepared_image"
            ]
        )

        chosen = dict(
            record
        )

        chosen[
            "smoke_test_reason"
        ] = reason

        selected.append(
            chosen
        )


def choose_positive_smoke_samples(
    records: list,
    rng: random.Random,
    annotation_index: dict,
) -> list:

    pool = records.copy()

    rng.shuffle(
        pool
    )

    pool = pool[
        :min(
            POSITIVE_PROFILE_POOL,
            len(
                pool
            ),
        )
    ]

    profiled = [
        profile_record(
            record,
            annotation_index,
        )
        for record
        in tqdm(
            pool,
            desc="Profiling smoke-positive pool",
            unit="img",
            dynamic_ncols=True,
        )
    ]

    selected = []
    selected_paths = set()

    selectors = [
        (
            lambda r:
                r[
                    "_tiny_count"
                ],
            "tiny_fish_heavy",
            True,
        ),

        (
            lambda r:
                r[
                    "_small_count"
                ],
            "small_fish_heavy",
            True,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "brightness"
                ],
            "dark_scene",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "brightness"
                ],
            "bright_scene",
            True,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "contrast"
                ],
            "low_contrast_scene",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "contrast"
                ],
            "high_contrast_scene",
            True,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "sharpness"
                ],
            "soft_scene",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "sharpness"
                ],
            "high_detail_scene",
            True,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "colour_cast"
                ],
            "strong_colour_cast",
            True,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "colour_cast"
                ],
            "weak_colour_cast",
            False,
        ),

        (
            lambda r:
                r[
                    "target_count"
                ],
            "many_fish",
            True,
        ),
    ]

    for (
        key_function,
        reason,
        reverse,
    ) in selectors:

        chosen = choose_ranked(
            profiled,
            selected_paths,
            key_function,
            reason,
            reverse,
        )

        if chosen is not None:
            selected.append(
                chosen
            )

    fill_random_samples(
        profiled,
        selected,
        selected_paths,
        SMOKE_POSITIVE_COUNT,
        rng,
        "diverse_positive_random",
    )

    return selected[
        :SMOKE_POSITIVE_COUNT
    ]


def choose_negative_smoke_samples(
    records: list,
    rng: random.Random,
    annotation_index: dict,
) -> list:

    pool = records.copy()

    rng.shuffle(
        pool
    )

    pool = pool[
        :min(
            NEGATIVE_PROFILE_POOL,
            len(
                pool
            ),
        )
    ]

    profiled = [
        profile_record(
            record,
            annotation_index,
        )
        for record
        in tqdm(
            pool,
            desc="Profiling smoke-negative pool",
            unit="img",
            dynamic_ncols=True,
        )
    ]

    selected = []
    selected_paths = set()

    selectors = [
        (
            lambda r:
                r[
                    "_profile"
                ][
                    "brightness"
                ],
            "negative_dark_scene",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "contrast"
                ],
            "negative_low_contrast",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "sharpness"
                ],
            "negative_soft_scene",
            False,
        ),

        (
            lambda r:
                r[
                    "_profile"
                ][
                    "colour_cast"
                ],
            "negative_strong_colour_cast",
            True,
        ),
    ]

    for (
        key_function,
        reason,
        reverse,
    ) in selectors:

        chosen = choose_ranked(
            profiled,
            selected_paths,
            key_function,
            reason,
            reverse,
        )

        if chosen is not None:
            selected.append(
                chosen
            )

    fill_random_samples(
        profiled,
        selected,
        selected_paths,
        SMOKE_NEGATIVE_COUNT,
        rng,
        "diverse_negative_random",
    )

    return selected[
        :SMOKE_NEGATIVE_COUNT
    ]


def choose_smoke_samples(
    train_records: list,
    annotation_index: dict,
) -> list:

    print_header(
        "INDEPENDENT SMOKE SAMPLE SELECTION"
    )

    positives = [
        record
        for record
        in train_records
        if record[
            "target_count"
        ]
        > 0
    ]

    negatives = [
        record
        for record
        in train_records
        if record[
            "target_count"
        ]
        == 0
    ]

    rng = random.Random(
        SMOKE_TEST_SEED
    )

    positive_samples = (
        choose_positive_smoke_samples(
            positives,
            rng,
            annotation_index,
        )
    )

    negative_samples = (
        choose_negative_smoke_samples(
            negatives,
            rng,
            annotation_index,
        )
    )

    samples = (
        positive_samples
        + negative_samples
    )

    print()

    print(
        f"Smoke positives         : "
        f"{len(positive_samples)}"
    )

    print(
        f"Smoke negatives         : "
        f"{len(negative_samples)}"
    )

    print(
        f"Smoke total             : "
        f"{len(samples)}"
    )

    return samples


# ============================================================
# v1.4.3 DEPLOYMENT-GROUP CALIBRATION SAMPLING
# ============================================================


def get_deployment(
    record: dict,
) -> str:

    deployment = record.get(
        "deployment"
    )

    if (
        deployment is None
        or not str(
            deployment
        ).strip()
    ):
        raise RuntimeError(
            "Prepared image record is missing "
            f"'deployment': {record.get('prepared_image')}"
        )

    return str(
        deployment
    )


def group_records_by_deployment(
    records: list,
) -> dict[
    str,
    list,
]:

    groups = defaultdict(
        list
    )

    for record in records:
        groups[
            get_deployment(
                record
            )
        ].append(
            record
        )

    return dict(
        groups
    )


def grouped_round_robin_select(
    groups: dict[
        str,
        list,
    ],
    desired_count: int,
    selected_paths: set,
    selected_per_deployment: Counter,
    rng: random.Random,
    predicate=None,
) -> list:
    """
    Select across deployment groups in deterministic shuffled rounds.

    A single selected_per_deployment counter is shared by every
    calibration stratum. Positive and negative selections therefore
    cannot collectively exceed the deployment cap.
    """

    if desired_count <= 0:
        return []

    deployment_names = list(
        groups.keys()
    )

    rng.shuffle(
        deployment_names
    )

    queues = {}

    for deployment in deployment_names:

        candidates = [
            record
            for record
            in groups[
                deployment
            ]
            if (
                record[
                    "prepared_image"
                ]
                not in selected_paths
                and (
                    predicate is None
                    or predicate(
                        record
                    )
                )
            )
        ]

        rng.shuffle(
            candidates
        )

        queues[
            deployment
        ] = candidates

    selected = []

    while (
        len(
            selected
        )
        < desired_count
    ):

        progress = False
        current_order = (
            deployment_names.copy()
        )

        rng.shuffle(
            current_order
        )

        for deployment in current_order:

            if (
                len(
                    selected
                )
                >= desired_count
            ):
                break

            if (
                selected_per_deployment[
                    deployment
                ]
                >= MAX_CALIBRATION_IMAGES_PER_DEPLOYMENT
            ):
                continue

            queue = queues[
                deployment
            ]

            while queue:

                record = queue.pop()
                path = record[
                    "prepared_image"
                ]

                if path in selected_paths:
                    continue

                selected.append(
                    record
                )

                selected_paths.add(
                    path
                )

                selected_per_deployment[
                    deployment
                ] += 1

                progress = True
                break

        if not progress:
            break

    return selected


def choose_calibration_samples(
    train_records: list,
    smoke_samples: list,
    annotation_index: dict,
) -> list:

    print_header(
        "TRAIN-ONLY GROUP-AWARE CALIBRATION SAMPLE SELECTION"
    )

    smoke_deployments = {
        get_deployment(
            record
        )
        for record
        in smoke_samples
    }

    eligible_records = [
        record
        for record
        in train_records
        if (
            get_deployment(
                record
            )
            not in smoke_deployments
        )
    ]

    excluded_by_deployment = (
        len(
            train_records
        )
        - len(
            eligible_records
        )
    )

    positives = []
    negatives = []

    for record in eligible_records:

        if record[
            "target_count"
        ] > 0:

            annotations = (
                annotation_index.get(
                    record[
                        "prepared_image"
                    ],
                    [],
                )
            )

            sizes = bbox_size_counts(
                annotations,
                record[
                    "width"
                ],
                record[
                    "height"
                ],
            )

            enriched = dict(
                record
            )

            enriched[
                "_tiny_count"
            ] = sizes[
                "tiny"
            ]

            enriched[
                "_small_count"
            ] = sizes[
                "small"
            ]

            positives.append(
                enriched
            )

        else:
            negatives.append(
                dict(
                    record
                )
            )

    positive_groups = (
        group_records_by_deployment(
            positives
        )
    )

    negative_groups = (
        group_records_by_deployment(
            negatives
        )
    )

    rng = random.Random(
        CALIBRATION_SEED
    )

    selected_paths = set()
    selected_per_deployment = Counter()
    selected_positive = []

    tiny_selection = (
        grouped_round_robin_select(
            groups=positive_groups,
            desired_count=(
                DESIRED_TINY_CALIBRATION_IMAGES
            ),
            selected_paths=selected_paths,
            selected_per_deployment=(
                selected_per_deployment
            ),
            rng=rng,
            predicate=lambda record: (
                record.get(
                    "_tiny_count",
                    0,
                )
                > 0
            ),
        )
    )

    selected_positive.extend(
        tiny_selection
    )

    remaining_positive_slots = (
        CALIBRATION_POSITIVE_COUNT
        - len(
            selected_positive
        )
    )

    small_selection = (
        grouped_round_robin_select(
            groups=positive_groups,
            desired_count=min(
                DESIRED_SMALL_CALIBRATION_IMAGES,
                max(
                    0,
                    remaining_positive_slots,
                ),
            ),
            selected_paths=selected_paths,
            selected_per_deployment=(
                selected_per_deployment
            ),
            rng=rng,
            predicate=lambda record: (
                record.get(
                    "_tiny_count",
                    0,
                )
                == 0
                and record.get(
                    "_small_count",
                    0,
                )
                > 0
            ),
        )
    )

    selected_positive.extend(
        small_selection
    )

    positive_fill = (
        grouped_round_robin_select(
            groups=positive_groups,
            desired_count=max(
                0,
                CALIBRATION_POSITIVE_COUNT
                - len(
                    selected_positive
                ),
            ),
            selected_paths=selected_paths,
            selected_per_deployment=(
                selected_per_deployment
            ),
            rng=rng,
        )
    )

    selected_positive.extend(
        positive_fill
    )

    if (
        len(
            selected_positive
        )
        != CALIBRATION_POSITIVE_COUNT
    ):
        raise RuntimeError(
            "Unable to create the required positive calibration "
            "set after excluding smoke deployments and applying "
            "the per-deployment cap. "
            f"Selected: {len(selected_positive)}"
        )

    selected_negative = (
        grouped_round_robin_select(
            groups=negative_groups,
            desired_count=(
                CALIBRATION_NEGATIVE_COUNT
            ),
            selected_paths=selected_paths,
            selected_per_deployment=(
                selected_per_deployment
            ),
            rng=rng,
        )
    )

    if (
        len(
            selected_negative
        )
        != CALIBRATION_NEGATIVE_COUNT
    ):
        raise RuntimeError(
            "Unable to create the required negative calibration "
            "set after excluding smoke deployments and applying "
            "the shared per-deployment cap. "
            f"Selected: {len(selected_negative)}"
        )

    calibration = (
        selected_positive
        + selected_negative
    )

    rng.shuffle(
        calibration
    )

    tiny_images = sum(
        1
        for record
        in selected_positive
        if record.get(
            "_tiny_count",
            0,
        )
        > 0
    )

    small_images = sum(
        1
        for record
        in selected_positive
        if record.get(
            "_small_count",
            0,
        )
        > 0
    )

    print(
        f"Smoke deployments held out: "
        f"{len(smoke_deployments)}"
    )

    print(
        f"TRAIN images held out     : "
        f"{excluded_by_deployment}"
    )

    print(
        f"Calibration positives     : "
        f"{len(selected_positive)}"
    )

    print(
        f"Calibration negatives     : "
        f"{len(selected_negative)}"
    )

    print(
        f"Calibration total         : "
        f"{len(calibration)}"
    )

    print(
        f"Positive images w/tiny    : "
        f"{tiny_images}"
    )

    print(
        f"Positive images w/small   : "
        f"{small_images}"
    )

    print(
        f"Deployment cap            : "
        f"{MAX_CALIBRATION_IMAGES_PER_DEPLOYMENT}"
    )

    return calibration


def sampling_independence_preflight(
    train_records: list,
    smoke_samples: list,
    calibration_samples: list,
) -> dict:

    print_header(
        "CALIBRATION / SMOKE DEPLOYMENT-INDEPENDENCE PREFLIGHT"
    )

    smoke_paths = {
        record[
            "prepared_image"
        ]
        for record
        in smoke_samples
    }

    calibration_paths = {
        record[
            "prepared_image"
        ]
        for record
        in calibration_samples
    }

    smoke_deployments = {
        get_deployment(
            record
        )
        for record
        in smoke_samples
    }

    calibration_deployments = {
        get_deployment(
            record
        )
        for record
        in calibration_samples
    }

    image_overlap = sorted(
        smoke_paths
        & calibration_paths
    )

    deployment_overlap = sorted(
        smoke_deployments
        & calibration_deployments
    )

    calibration_counts = Counter(
        get_deployment(
            record
        )
        for record
        in calibration_samples
    )

    cap_violations = {
        deployment: count
        for deployment, count
        in sorted(
            calibration_counts.items()
        )
        if (
            count
            > MAX_CALIBRATION_IMAGES_PER_DEPLOYMENT
        )
    }

    maximum_observed = max(
        calibration_counts.values(),
        default=0,
    )

    excluded_train_images = sum(
        1
        for record
        in train_records
        if (
            get_deployment(
                record
            )
            in smoke_deployments
        )
    )

    report = {
        "generated_at_utc":
            utc_now(),

        "policy_version":
            PREPROCESSING_POLICY_VERSION,

        "status":
            "PASS",

        "policy": {
            "smoke_deployments_excluded_from_calibration":
                True,

            "group_aware_round_robin":
                True,

            "maximum_calibration_images_per_deployment":
                MAX_CALIBRATION_IMAGES_PER_DEPLOYMENT,

            "positive_and_negative_share_deployment_cap":
                True,

            "tiny_calibration_target":
                DESIRED_TINY_CALIBRATION_IMAGES,

            "small_calibration_target":
                DESIRED_SMALL_CALIBRATION_IMAGES,
        },

        "counts": {
            "smoke_images":
                len(smoke_samples),

            "smoke_deployments":
                len(smoke_deployments),

            "calibration_images":
                len(calibration_samples),

            "calibration_deployments":
                len(calibration_deployments),

            "train_images_excluded_by_smoke_deployment":
                excluded_train_images,

            "maximum_observed_calibration_images_per_deployment":
                maximum_observed,
        },

        "checks": {
            "image_overlap_count":
                len(image_overlap),

            "image_overlap":
                image_overlap,

            "image_overlap_passed":
                not image_overlap,

            "deployment_overlap_count":
                len(deployment_overlap),

            "deployment_overlap":
                deployment_overlap,

            "deployment_overlap_passed":
                not deployment_overlap,

            "deployment_cap_violations":
                cap_violations,

            "deployment_cap_passed":
                not cap_violations,
        },

        "smoke_deployments":
            sorted(
                smoke_deployments
            ),

        "calibration_deployment_counts":
            dict(
                sorted(
                    calibration_counts.items()
                )
            ),
    }

    failures = []

    if image_overlap:
        failures.append(
            "exact image overlap"
        )

    if deployment_overlap:
        failures.append(
            "deployment overlap"
        )

    if cap_violations:
        failures.append(
            "deployment cap violation"
        )

    if failures:
        report[
            "status"
        ] = "FAIL"

        raise RuntimeError(
            "Sampling independence preflight failed: "
            + ", ".join(
                failures
            )
        )

    print(
        f"Smoke images              : "
        f"{len(smoke_samples)}"
    )

    print(
        f"Smoke deployments         : "
        f"{len(smoke_deployments)}"
    )

    print(
        f"Calibration images        : "
        f"{len(calibration_samples)}"
    )

    print(
        f"Calibration deployments   : "
        f"{len(calibration_deployments)}"
    )

    print(
        "Exact image overlap       : 0"
    )

    print(
        "Deployment overlap        : 0"
    )

    print(
        f"Maximum per deployment    : "
        f"{maximum_observed}"
    )

    print(
        "Result                    : PASS"
    )

    return report


# ============================================================
# RESET OUTPUTS
# ============================================================


def reset_outputs() -> None:

    for directory in (
        CANDIDATES_ROOT,
        WINNERS_ROOT,
        GRIDS_ROOT,
        REPORTS_ROOT,
    ):

        if directory.exists():
            shutil.rmtree(
                directory
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# CALIBRATION EVALUATION
# ============================================================


def compact_candidate_summary(
    evaluation: dict,
    roi_audit: dict,
) -> dict:

    return {
        "risk_class":
            evaluation[
                "risk_class"
            ],

        "runtime_gate_passed":
            evaluation[
                "damage_gate_passed"
            ],

        "runtime_rejection_reasons":
            evaluation[
                "rejection_reasons"
            ],

        "benefit_score":
            float(
                evaluation[
                    "benefit_score"
                ]
            ),

        "improvement_score":
            float(
                evaluation[
                    "benefit_score"
                ]
            ),

        "safety_margin_score":
            float(
                evaluation[
                    "safety_margin_score"
                ]
            ),

        "benefit_components":
            evaluation[
                "benefit_components"
            ],

        "offline_roi_audit_applicable":
            roi_audit[
                "applicable"
            ],

        "offline_roi_audit_passed":
            roi_audit[
                "passed"
            ],

        "offline_roi_reasons":
            roi_audit[
                "reasons"
            ],
    }


def evaluate_calibration_set(
    calibration_samples: list,
    annotation_index: dict,
) -> list:

    print_header(
        "v1.4.3 LABEL-FREE CALIBRATION EVALUATION"
    )

    rows = []

    for record in tqdm(
        calibration_samples,
        desc="Evaluating calibration set",
        unit="img",
        dynamic_ncols=True,
    ):

        image_path = (
            PREPARED_ROOT
            / record[
                "prepared_image"
            ]
        )

        raw = cv2.imread(
            str(
                image_path
            )
        )

        if raw is None:
            raise RuntimeError(
                f"Unable to read: "
                f"{image_path}"
            )

        annotations = (
            annotation_index.get(
                record[
                    "prepared_image"
                ],
                [],
            )
        )

        (
            candidates,
            evaluations,
        ) = evaluate_all_candidates(
            raw
        )

        candidate_summaries = {}

        for mode in PREPROCESSING_MODES:

            if mode == "raw_baseline":

                candidate_summaries[
                    mode
                ] = {
                    "risk_class":
                        "baseline",

                    "runtime_gate_passed":
                        True,

                    "runtime_rejection_reasons":
                        [],

                    "benefit_score":
                        0.0,

                    "improvement_score":
                        0.0,

                    "safety_margin_score":
                        1.0,

                    "benefit_components":
                        {},

                    "offline_roi_audit_applicable":
                        bool(
                            annotations
                        ),

                    "offline_roi_audit_passed":
                        True,

                    "offline_roi_reasons":
                        [],
                }

                continue

            roi_audit = (
                offline_roi_audit(
                    raw,
                    candidates[
                        mode
                    ],
                    annotations,
                )
            )

            candidate_summaries[
                mode
            ] = (
                compact_candidate_summary(
                    evaluations[
                        mode
                    ],
                    roi_audit,
                )
            )

        rows.append(
            {
                "prepared_image":
                    record[
                        "prepared_image"
                    ],

                # Kept only for diagnostics.
                # NEVER used by threshold selection.
                "sample_type":
                    (
                        "positive"
                        if record[
                            "target_count"
                        ]
                        > 0
                        else "negative"
                    ),

                "target_count":
                    record[
                        "target_count"
                    ],

                "candidate_summaries":
                    candidate_summaries,
            }
        )

    return rows


# ============================================================
# COMPACT WINNER SELECTION
# ============================================================


def choose_compact_winner(
    candidate_summaries: dict,
    minimum_improvement: float,
) -> str:

    evaluations = {}

    for mode, summary in (
        candidate_summaries.items()
    ):

        evaluations[
            mode
        ] = {
            "damage_gate_passed":
                summary[
                    "runtime_gate_passed"
                ],

            "benefit_score":
                summary[
                    "benefit_score"
                ],

            "safety_margin_score":
                summary[
                    "safety_margin_score"
                ],
        }

    return (
        choose_winner_from_evaluations(
            evaluations,
            minimum_improvement,
        )
    )


# ============================================================
# v1.4.3 CLASS-NEUTRAL THRESHOLD EVALUATION
#
# IMPORTANT:
#
# Positive/negative status does NOT contribute to threshold
# quality.
#
# Fish labels are used ONLY for ROI safety auditing.
# ============================================================


def evaluate_threshold(
    calibration_rows: list,
    threshold: float,
) -> dict:

    total_images = len(
        calibration_rows
    )

    total_processed = 0

    selected_scores = []
    selected_margins = []

    roi_audited_selected = 0
    roi_unsafe_selected = 0

    winner_counts = Counter()

    # Diagnostic only.
    diagnostic_positive_processed = 0
    diagnostic_negative_processed = 0

    for row in calibration_rows:

        winner = (
            choose_compact_winner(
                row[
                    "candidate_summaries"
                ],
                threshold,
            )
        )

        winner_counts[
            winner
        ] += 1

        if winner == "raw_baseline":
            continue

        total_processed += 1

        winner_summary = (
            row[
                "candidate_summaries"
            ][
                winner
            ]
        )

        selected_scores.append(
            float(
                winner_summary[
                    "benefit_score"
                ]
            )
        )

        selected_margins.append(
            float(
                winner_summary[
                    "safety_margin_score"
                ]
            )
        )

        if row[
            "sample_type"
        ] == "positive":

            diagnostic_positive_processed += 1

        else:

            diagnostic_negative_processed += 1

        if winner_summary[
            "offline_roi_audit_applicable"
        ]:

            roi_audited_selected += 1

            if not winner_summary[
                "offline_roi_audit_passed"
            ]:

                roi_unsafe_selected += 1

    processed_rate = (
        total_processed
        / max(
            total_images,
            1,
        )
    )

    mean_selected_benefit = (
        float(
            np.mean(
                selected_scores
            )
        )
        if selected_scores
        else 0.0
    )

    median_selected_benefit = (
        float(
            np.median(
                selected_scores
            )
        )
        if selected_scores
        else 0.0
    )

    minimum_selected_benefit = (
        float(
            np.min(
                selected_scores
            )
        )
        if selected_scores
        else 0.0
    )

    mean_selected_safety_margin = (
        float(
            np.mean(
                selected_margins
            )
        )
        if selected_margins
        else 0.0
    )

    minimum_selected_safety_margin = (
        float(
            np.min(
                selected_margins
            )
        )
        if selected_margins
        else 0.0
    )

    enough_coverage = (
        total_processed
        >= MIN_CALIBRATION_PROCESSED_COUNT
        and processed_rate
        >= MIN_CALIBRATION_PROCESSED_RATE
    )

    zero_roi_damage = (
        roi_unsafe_selected
        == 0
    )

    eligible_for_selection = (
        total_processed > 0
        and enough_coverage
        and zero_roi_damage
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "total_images":
            total_images,

        "total_processed":
            total_processed,

        "processed_rate":
            float(
                processed_rate
            ),

        "mean_selected_benefit":
            mean_selected_benefit,

        "median_selected_benefit":
            median_selected_benefit,

        "minimum_selected_benefit":
            minimum_selected_benefit,

        "mean_selected_safety_margin":
            mean_selected_safety_margin,

        "minimum_selected_safety_margin":
            minimum_selected_safety_margin,

        "roi_audited_selected":
            roi_audited_selected,

        "roi_unsafe_selected":
            roi_unsafe_selected,

        "zero_roi_damage":
            zero_roi_damage,

        "enough_coverage":
            enough_coverage,

        "eligible_for_selection":
            eligible_for_selection,

        # Diagnostics only.
        # These values DO NOT affect calibration.
        "diagnostic_positive_processed":
            diagnostic_positive_processed,

        "diagnostic_negative_processed":
            diagnostic_negative_processed,

        "winner_counts":
            dict(
                winner_counts
            ),
    }


# ============================================================
# v1.4.3 THRESHOLD CALIBRATION
#
# Selection principle:
#
# Choose the LOWEST threshold which:
#
#   1. produces a meaningful number of transformations
#   2. has ZERO known fish-detail failures
#
# Why lowest?
#
# Once safety is satisfied, choosing the lowest safe evidence
# threshold gives the selector maximum opportunity to apply
# useful enhancement without using class labels.
# ============================================================


def calibrate_threshold(
    calibration_rows: list,
) -> tuple[
    float,
    dict,
]:

    print_header(
        "v1.4.3 CLASS-NEUTRAL THRESHOLD CALIBRATION"
    )

    analyses = [
        evaluate_threshold(
            calibration_rows,
            threshold,
        )
        for threshold
        in THRESHOLD_GRID
    ]

    eligible = [
        row
        for row
        in analyses
        if row[
            "eligible_for_selection"
        ]
    ]

    calibration_safe = False
    calibration_warning = None

    if eligible:

        # Lowest benefit threshold that achieves:
        # - meaningful coverage
        # - zero known ROI damage
        #
        # No class reward / class penalty.

        selected_row = min(
            eligible,
            key=lambda row:
                row[
                    "threshold"
                ],
        )

        selected_threshold = float(
            selected_row[
                "threshold"
            ]
        )

        calibration_safe = True

        method = (
            "lowest_zero_roi_damage_"
            "class_neutral_threshold"
        )

    else:

        # ----------------------------------------------------
        # Try zero-damage thresholds with low coverage.
        # We still report them, but policy is not safe to
        # freeze because calibration evidence is too weak.
        # ----------------------------------------------------

        zero_damage_nonempty = [
            row
            for row
            in analyses
            if (
                row[
                    "total_processed"
                ]
                > 0
                and row[
                    "roi_unsafe_selected"
                ]
                == 0
            )
        ]

        if zero_damage_nonempty:

            selected_row = min(
                zero_damage_nonempty,
                key=lambda row:
                    row[
                        "threshold"
                    ],
            )

            selected_threshold = float(
                selected_row[
                    "threshold"
                ]
            )

            method = (
                "zero_roi_damage_but_"
                "insufficient_calibration_coverage"
            )

            calibration_warning = (
                "A zero-ROI-damage threshold exists, "
                "but it processes too few calibration images "
                "to provide strong evidence."
            )

        else:

            # No acceptable zero-damage threshold.
            #
            # Keep a deterministic fallback for diagnostics,
            # but explicitly mark calibration unsafe.

            selected_threshold = (
                DEFAULT_MINIMUM_IMPROVEMENT
            )

            method = (
                "unsafe_fallback_default_threshold"
            )

            calibration_warning = (
                "No tested threshold achieved both "
                "non-empty selection and zero fish-ROI damage."
            )

    print(
        f"Selected threshold      : "
        f"{selected_threshold:.4f}"
    )

    print(
        f"Calibration safe        : "
        f"{calibration_safe}"
    )

    print(
        f"Calibration method      : "
        f"{method}"
    )

    if calibration_warning:

        print(
            f"Warning                 : "
            f"{calibration_warning}"
        )

    print()
    print(
        "Threshold analysis:"
    )

    for row in analyses:

        marker = (
            " <-- SELECTED"
            if math.isclose(
                row[
                    "threshold"
                ],
                selected_threshold,
                abs_tol=1e-12,
            )
            else ""
        )

        eligible_marker = (
            " ELIGIBLE"
            if row[
                "eligible_for_selection"
            ]
            else ""
        )

        print(
            f"  {row['threshold']:.4f}"
            f" | processed={row['total_processed']:>3}"
            f" ({row['processed_rate'] * 100:>5.1f}%)"
            f" | roi_bad={row['roi_unsafe_selected']:>2}"
            f" | benefit={row['mean_selected_benefit']:+.5f}"
            f" | margin={row['mean_selected_safety_margin']:.3f}"
            f"{eligible_marker}"
            f"{marker}"
        )

    report = {
        "policy_version":
            PREPROCESSING_POLICY_VERSION,

        "method":
            method,

        "scope":
            "train_only",

        "calibration_safe":
            calibration_safe,

        "calibration_warning":
            calibration_warning,

        "runtime_selector":
            "label_free",

        "score_model":
            "benefit_only_after_safety_gate",

        "threshold_selection":
            (
                "lowest threshold with meaningful "
                "class-neutral coverage and zero "
                "selected fish-ROI failures"
            ),

        "class_labels_used_for_threshold_reward":
            False,

        "negative_images_penalized":
            False,

        "positive_images_rewarded":
            False,

        "annotations_used_for":
            [
                "offline fish ROI safety audit",
                "tiny-fish safety audit",
                "representative calibration sampling",
            ],

        "annotations_used_for_runtime_selection":
            False,

        "benefit_weights":
            BENEFIT_WEIGHTS,

        "minimum_calibration_processed_count":
            MIN_CALIBRATION_PROCESSED_COUNT,

        "minimum_calibration_processed_rate":
            MIN_CALIBRATION_PROCESSED_RATE,

        "selected_threshold":
            selected_threshold,

        "threshold_grid":
            list(
                THRESHOLD_GRID
            ),

        "threshold_analysis":
            analyses,

        "calibration_rows":
            calibration_rows,
    }

    return (
        selected_threshold,
        report,
    )


# ============================================================
# VISUAL BOX OVERLAY
# ============================================================


def draw_ground_truth_boxes(
    image: np.ndarray,
    annotations: list,
) -> np.ndarray:

    output = image.copy()

    height, width = (
        output.shape[:2]
    )

    thickness = max(
        2,
        int(
            round(
                min(
                    width,
                    height,
                )
                / 300.0
            )
        ),
    )

    for annotation in annotations:

        bbox = annotation.get(
            "bbox_xywh"
        )

        if (
            not isinstance(
                bbox,
                list,
            )
            or len(
                bbox
            )
            != 4
        ):
            continue

        (
            x,
            y,
            box_width,
            box_height,
        ) = map(
            float,
            bbox,
        )

        x1 = max(
            0,
            min(
                width - 1,
                int(
                    round(
                        x
                    )
                ),
            ),
        )

        y1 = max(
            0,
            min(
                height - 1,
                int(
                    round(
                        y
                    )
                ),
            ),
        )

        x2 = max(
            0,
            min(
                width - 1,
                int(
                    round(
                        x
                        + box_width
                    )
                ),
            ),
        )

        y2 = max(
            0,
            min(
                height - 1,
                int(
                    round(
                        y
                        + box_height
                    )
                ),
            ),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue

        size_class = (
            classify_bbox_size(
                bbox,
                width,
                height,
            )
        )

        if size_class == "tiny":
            label = "fish:TINY"

        elif size_class == "small":
            label = "fish:small"

        else:
            label = "fish"

        colour = (
            0,
            255,
            255,
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            colour,
            thickness,
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    15,
                    y1 - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# COMPARISON GRID
# ============================================================


def make_thumbnail(
    image: np.ndarray,
    label: str,
    width: int = 420,
    height: int = 300,
) -> np.ndarray:

    canvas = np.zeros(
        (
            height + 44,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    image_height, image_width = (
        image.shape[:2]
    )

    scale = min(
        width
        / image_width,
        height
        / image_height,
    )

    resized_width = max(
        1,
        int(
            image_width
            * scale
        ),
    )

    resized_height = max(
        1,
        int(
            image_height
            * scale
        ),
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        ),
    )

    x_offset = (
        width
        - resized_width
    ) // 2

    y_offset = (
        height
        - resized_height
    ) // 2

    canvas[
        y_offset:
            y_offset
            + resized_height,

        x_offset:
            x_offset
            + resized_width,
    ] = resized

    cv2.putText(
        canvas,
        label,
        (
            10,
            height + 29,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    return canvas


def create_comparison_grid(
    candidates: dict,
    evaluations: dict,
    winner_mode: str,
    annotations: list,
) -> np.ndarray:

    thumbnails = []

    for mode in PREPROCESSING_MODES:

        evaluation = (
            evaluations[
                mode
            ]
        )

        score = float(
            evaluation[
                "benefit_score"
            ]
        )

        if mode == winner_mode:
            marker = "WINNER"

        elif evaluation[
            "damage_gate_passed"
        ]:
            marker = "PASS"

        else:
            marker = "REJECT"

        if mode == "raw_baseline":

            risk = "BASE"

        elif mode in HIGHER_RISK_MODES:

            risk = "HIGH"

        else:

            risk = "NORMAL"

        label = (
            f"{mode} | "
            f"{risk} | "
            f"{marker} | "
            f"{score:+.4f}"
        )

        visual = (
            draw_ground_truth_boxes(
                candidates[
                    mode
                ],
                annotations,
            )
        )

        thumbnails.append(
            make_thumbnail(
                visual,
                label,
            )
        )

    row_1 = cv2.hconcat(
        thumbnails[
            0:4
        ]
    )

    row_2 = cv2.hconcat(
        thumbnails[
            4:8
        ]
    )

    return cv2.vconcat(
        (
            row_1,
            row_2,
        )
    )


# ============================================================
# SAVE SMOKE OUTPUTS
# ============================================================


def save_smoke_outputs(
    sample_number: int,
    record: dict,
    winner_mode: str,
    winner_image: np.ndarray,
    candidates: dict,
    evaluations: dict,
    annotations: list,
) -> None:

    sample_id = (
        f"{sample_number:02d}"
    )

    source_name = Path(
        record[
            "prepared_image"
        ]
    ).name

    candidate_directory = (
        CANDIDATES_ROOT
        / sample_id
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode in PREPROCESSING_MODES:

        output_path = (
            candidate_directory
            / f"{mode}.jpg"
        )

        if not cv2.imwrite(
            str(
                output_path
            ),
            candidates[
                mode
            ],
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        ):
            raise RuntimeError(
                f"Failed to save: "
                f"{output_path}"
            )

    winner_path = (
        WINNERS_ROOT
        / (
            f"{sample_id}"
            f"__{winner_mode}"
            f"__{source_name}"
        )
    )

    if not cv2.imwrite(
        str(
            winner_path
        ),
        winner_image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    ):
        raise RuntimeError(
            f"Failed to save: "
            f"{winner_path}"
        )

    grid = create_comparison_grid(
        candidates,
        evaluations,
        winner_mode,
        annotations,
    )

    grid_path = (
        GRIDS_ROOT
        / (
            f"{sample_id}"
            f"__{source_name}"
        )
    )

    if not cv2.imwrite(
        str(
            grid_path
        ),
        grid,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    ):
        raise RuntimeError(
            f"Failed to save: "
            f"{grid_path}"
        )


# ============================================================
# INDEPENDENT SMOKE TEST
# ============================================================


def run_smoke_test(
    smoke_samples: list,
    annotation_index: dict,
    minimum_improvement: float,
    calibration_safe: bool,
) -> dict:

    print_header(
        "INDEPENDENT v1.4.3 LABEL-FREE SMOKE TEST"
    )

    decisions = []

    winner_counts = Counter()
    runtime_rejection_counts = Counter()
    roi_failure_counts = Counter()

    higher_risk_winners = 0

    processed_positive_winners = 0
    processed_negative_winners = 0

    winner_roi_failures = 0

    for sample_number, record in enumerate(
        tqdm(
            smoke_samples,
            desc="Smoke testing",
            unit="img",
            dynamic_ncols=True,
        ),
        start=1,
    ):

        image_path = (
            PREPARED_ROOT
            / record[
                "prepared_image"
            ]
        )

        raw = cv2.imread(
            str(
                image_path
            )
        )

        if raw is None:
            raise RuntimeError(
                f"Unable to read: "
                f"{image_path}"
            )

        annotations = (
            annotation_index.get(
                record[
                    "prepared_image"
                ],
                [],
            )
        )

        (
            winner_mode,
            winner_image,
            evaluations,
            candidates,
        ) = select_best_candidate(
            raw,
            minimum_improvement,
        )

        winner_counts[
            winner_mode
        ] += 1

        if winner_mode in HIGHER_RISK_MODES:

            higher_risk_winners += 1

        processed = (
            winner_mode
            != "raw_baseline"
        )

        if processed:

            if record[
                "target_count"
            ] > 0:

                processed_positive_winners += 1

            else:

                processed_negative_winners += 1

        for mode in PREPROCESSING_MODES:

            if mode == "raw_baseline":
                continue

            evaluation = (
                evaluations[
                    mode
                ]
            )

            if not evaluation[
                "damage_gate_passed"
            ]:

                for reason in (
                    evaluation[
                        "rejection_reasons"
                    ]
                ):

                    runtime_rejection_counts[
                        reason
                    ] += 1

        winner_roi_audit = (
            offline_roi_audit(
                raw,
                winner_image,
                annotations,
            )
        )

        if (
            processed
            and winner_roi_audit[
                "applicable"
            ]
            and not winner_roi_audit[
                "passed"
            ]
        ):

            winner_roi_failures += 1

            for reason in (
                winner_roi_audit[
                    "reasons"
                ]
            ):

                roi_failure_counts[
                    reason
                ] += 1

        sizes = bbox_size_counts(
            annotations,
            record[
                "width"
            ],
            record[
                "height"
            ],
        )

        decision = {
            "sample_number":
                sample_number,

            "sample_type":
                (
                    "positive"
                    if record[
                        "target_count"
                    ]
                    > 0
                    else "negative"
                ),

            "selection_reason":
                record[
                    "smoke_test_reason"
                ],

            "prepared_image":
                record[
                    "prepared_image"
                ],

            "target_count":
                record[
                    "target_count"
                ],

            "tiny_count":
                sizes[
                    "tiny"
                ],

            "small_count":
                sizes[
                    "small"
                ],

            "medium_large_count":
                sizes[
                    "medium_large"
                ],

            "winner_mode":
                winner_mode,

            "winner_risk":
                (
                    "higher"
                    if winner_mode
                    in HIGHER_RISK_MODES
                    else (
                        "baseline"
                        if winner_mode
                        == "raw_baseline"
                        else "normal"
                    )
                ),

            "winner_benefit_score":
                float(
                    evaluations[
                        winner_mode
                    ][
                        "benefit_score"
                    ]
                ),

            "winner_safety_margin":
                float(
                    evaluations[
                        winner_mode
                    ][
                        "safety_margin_score"
                    ]
                ),

            "processed_selected":
                processed,

            "winner_offline_roi_audit":
                winner_roi_audit,

            "candidate_evaluations":
                evaluations,
        }

        decisions.append(
            decision
        )

        save_smoke_outputs(
            sample_number,
            record,
            winner_mode,
            winner_image,
            candidates,
            evaluations,
            annotations,
        )

    raw_winners = (
        winner_counts[
            "raw_baseline"
        ]
    )

    processed_winners = (
        len(
            smoke_samples
        )
        - raw_winners
    )

    if not calibration_safe:

        status = (
            "CALIBRATION_NOT_APPROVABLE"
        )

    elif winner_roi_failures > 0:

        status = (
            "DETAIL_AUDIT_FAILED"
        )

    else:

        status = (
            "REVIEW_REQUIRED"
        )

    winner_distribution = {}

    for mode in PREPROCESSING_MODES:

        count = (
            winner_counts[
                mode
            ]
        )

        winner_distribution[
            mode
        ] = {
            "count":
                count,

            "percentage":
                round(
                    count
                    / len(
                        smoke_samples
                    )
                    * 100.0,
                    2,
                ),
        }

    return {
        "status":
            status,

        "generated_at_utc":
            utc_now(),

        "policy_version":
            PREPROCESSING_POLICY_VERSION,

        "runtime_selector_label_free":
            True,

        "annotations_used_for_runtime_selection":
            False,

        "class_neutral_threshold_calibration":
            True,

        "score_model":
            "benefit_only_after_safety_gate",

        "calibrated_minimum_improvement":
            minimum_improvement,

        "sample_configuration": {
            "total":
                len(
                    smoke_samples
                ),

            "positive":
                sum(
                    1
                    for record
                    in smoke_samples
                    if record[
                        "target_count"
                    ]
                    > 0
                ),

            "negative":
                sum(
                    1
                    for record
                    in smoke_samples
                    if record[
                        "target_count"
                    ]
                    == 0
                ),
        },

        "raw_winners":
            raw_winners,

        "processed_winners":
            processed_winners,

        # Diagnostics only.
        "processed_positive_winners":
            processed_positive_winners,

        "processed_negative_winners":
            processed_negative_winners,

        "higher_risk_winners":
            higher_risk_winners,

        "winner_distribution":
            winner_distribution,

        "runtime_rejection_reason_counts":
            dict(
                runtime_rejection_counts
            ),

        "winner_roi_audit_failures":
            winner_roi_failures,

        "winner_roi_failure_reason_counts":
            dict(
                roi_failure_counts
            ),

        "decisions":
            decisions,
    }


# ============================================================
# DECISION CSV
# ============================================================


def write_decision_csv(
    decisions: list,
) -> None:

    path = (
        REPORTS_ROOT
        / "decisions.csv"
    )

    fields = [
        "sample_number",
        "sample_type",
        "selection_reason",
        "prepared_image",
        "target_count",
        "tiny_count",
        "small_count",
        "medium_large_count",
        "winner_mode",
        "winner_risk",
        "winner_benefit_score",
        "winner_safety_margin",
        "processed_selected",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for decision in decisions:

            writer.writerow(
                {
                    field:
                        decision[
                            field
                        ]
                    for field
                    in fields
                }
            )


# ============================================================
# MODE SAFETY REPORT
# ============================================================


def calibration_mode_report(
    calibration_rows: list,
) -> dict:

    result = {}

    for mode in PREPROCESSING_MODES:

        if mode == "raw_baseline":
            continue

        runtime_passes = 0
        runtime_rejections = 0

        roi_passes = 0
        roi_failures = 0

        runtime_reasons = Counter()
        roi_reasons = Counter()

        benefit_scores = []

        for row in calibration_rows:

            summary = (
                row[
                    "candidate_summaries"
                ][
                    mode
                ]
            )

            if summary[
                "runtime_gate_passed"
            ]:

                runtime_passes += 1

                benefit_scores.append(
                    float(
                        summary[
                            "benefit_score"
                        ]
                    )
                )

            else:

                runtime_rejections += 1

                for reason in (
                    summary[
                        "runtime_rejection_reasons"
                    ]
                ):

                    runtime_reasons[
                        reason
                    ] += 1

            if summary[
                "offline_roi_audit_applicable"
            ]:

                if summary[
                    "offline_roi_audit_passed"
                ]:

                    roi_passes += 1

                else:

                    roi_failures += 1

                    for reason in (
                        summary[
                            "offline_roi_reasons"
                        ]
                    ):

                        roi_reasons[
                            reason
                        ] += 1

        if benefit_scores:

            benefit_array = np.asarray(
                benefit_scores,
                dtype=np.float64,
            )

            benefit_statistics = {
                "min":
                    float(
                        np.min(
                            benefit_array
                        )
                    ),

                "mean":
                    float(
                        np.mean(
                            benefit_array
                        )
                    ),

                "median":
                    float(
                        np.median(
                            benefit_array
                        )
                    ),

                "p90":
                    float(
                        np.percentile(
                            benefit_array,
                            90,
                        )
                    ),

                "max":
                    float(
                        np.max(
                            benefit_array
                        )
                    ),
            }

        else:

            benefit_statistics = None

        result[
            mode
        ] = {
            "risk_class":
                (
                    "higher"
                    if mode
                    in HIGHER_RISK_MODES
                    else "normal"
                ),

            "runtime_passes":
                runtime_passes,

            "runtime_rejections":
                runtime_rejections,

            "runtime_rejection_reasons":
                dict(
                    runtime_reasons
                ),

            "offline_roi_passes":
                roi_passes,

            "offline_roi_failures":
                roi_failures,

            "offline_roi_failure_reasons":
                dict(
                    roi_reasons
                ),

            "safe_benefit_statistics":
                benefit_statistics,
        }

    return result


# ============================================================
# POLICY CANDIDATE
# ============================================================


def create_policy_candidate(
    minimum_improvement: float,
    calibration_report: dict,
    dataset_info: dict,
) -> dict:

    policy = {
        "policy_version":
            PREPROCESSING_POLICY_VERSION,

        "approval_status":
            (
                "REVIEW_REQUIRED"
                if calibration_report[
                    "calibration_safe"
                ]
                else "CALIBRATION_NOT_APPROVABLE"
            ),

        "created_at_utc":
            utc_now(),

        "dataset_fingerprint_sha256":
            dataset_info.get(
                "dataset_fingerprint_sha256"
            ),

        "runtime_selector":
            "label_free",

        "annotations_used_at_runtime":
            False,

        "class_neutral_threshold_calibration":
            True,

        "positive_images_rewarded":
            False,

        "negative_images_penalized":
            False,

        "annotations_used_for": [
            "offline fish ROI safety validation",
            "tiny-fish safety validation",
            "representative calibration sampling",
            "visual comparison grids",
        ],

        "score_model":
            "benefit_only_after_safety_gate",

        "benefit_weights":
            BENEFIT_WEIGHTS,

        "minimum_improvement":
            minimum_improvement,

        "raw_fallback":
            True,

        "candidate_modes":
            list(
                PREPROCESSING_MODES
            ),

        "normal_risk_modes":
            sorted(
                NORMAL_RISK_MODES
            ),

        "higher_risk_modes":
            sorted(
                HIGHER_RISK_MODES
            ),

        "higher_risk_extra_improvement":
            HIGH_RISK_EXTRA_IMPROVEMENT,

        "higher_risk_advantage_over_safe":
            HIGH_RISK_ADVANTAGE_OVER_SAFE,

        "benefit_tie_epsilon":
            BENEFIT_TIE_EPSILON,

        "forbidden_operations":
            list(
                FORBIDDEN_OPERATIONS
            ),

        "local_detail_protection": {
            "grid_sizes":
                list(
                    LOCAL_GRID_SIZES
                ),

            "meaningful_tile_keep_ratio":
                LOCAL_MEANINGFUL_TILE_KEEP_RATIO,

            "min_detail_p10":
                MIN_LOCAL_DETAIL_P10,

            "min_edge_recall_p10":
                MIN_LOCAL_EDGE_RECALL_P10,

            "min_sharpness_p10":
                MIN_LOCAL_SHARPNESS_P10,

            "max_sharpness_p90":
                MAX_LOCAL_SHARPNESS_P90,

            "max_noise_p90":
                MAX_LOCAL_NOISE_P90,
        },

        "calibration": {
            "scope":
                "train_only",

            "sample_count":
                CALIBRATION_TOTAL_COUNT,

            "smoke_overlap":
                0,

            "method":
                calibration_report[
                    "method"
                ],

            "calibration_safe":
                calibration_report[
                    "calibration_safe"
                ],

            "threshold_grid":
                calibration_report[
                    "threshold_grid"
                ],

            "minimum_processed_count":
                MIN_CALIBRATION_PROCESSED_COUNT,

            "minimum_processed_rate":
                MIN_CALIBRATION_PROCESSED_RATE,
        },
    }

    canonical = json.dumps(
        policy,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    policy[
        "policy_fingerprint_sha256"
    ] = hashlib.sha256(
        canonical
    ).hexdigest()

    return policy


# ============================================================
# SUMMARY
# ============================================================


def print_summary(
    smoke_report: dict,
    calibration_report: dict,
) -> None:

    print_header(
        "v1.4.3 SMOKE TEST SUMMARY"
    )

    configuration = (
        smoke_report[
            "sample_configuration"
        ]
    )

    print(
        f"Status                    : "
        f"{smoke_report['status']}"
    )

    print(
        f"Policy version            : "
        f"{smoke_report['policy_version']}"
    )

    print(
        "Runtime selector          : LABEL-FREE"
    )

    print(
        "Annotations at runtime    : NO"
    )

    print(
        "Threshold class-neutral   : YES"
    )

    print(
        "Positive reward           : NO"
    )

    print(
        "Negative penalty          : NO"
    )

    print(
        "Score architecture        : "
        "SAFETY GATE → BENEFIT SCORE"
    )

    print()

    print(
        f"Candidate modes           : "
        f"{len(PREPROCESSING_MODES)}"
    )

    print(
        f"Normal-risk modes         : "
        f"{len(NORMAL_RISK_MODES)}"
    )

    print(
        f"Higher-risk modes         : "
        f"{len(HIGHER_RISK_MODES)}"
    )

    print()

    print(
        f"Calibration images        : "
        f"{CALIBRATION_TOTAL_COUNT}"
    )

    print(
        f"Calibration threshold     : "
        f"{smoke_report['calibrated_minimum_improvement']:.4f}"
    )

    print(
        f"Calibration method        : "
        f"{calibration_report['method']}"
    )

    print(
        f"Calibration safe          : "
        f"{calibration_report['calibration_safe']}"
    )

    print()

    print(
        f"Smoke samples             : "
        f"{configuration['total']}"
    )

    print(
        f"Fish-positive             : "
        f"{configuration['positive']}"
    )

    print(
        f"Negative/background       : "
        f"{configuration['negative']}"
    )

    print()

    print(
        f"Raw winners               : "
        f"{smoke_report['raw_winners']}"
    )

    print(
        f"Processed winners         : "
        f"{smoke_report['processed_winners']}"
    )

    # Diagnostic only.
    print(
        f"Processed positive        : "
        f"{smoke_report['processed_positive_winners']}"
    )

    print(
        f"Processed negative        : "
        f"{smoke_report['processed_negative_winners']}"
    )

    print(
        f"Higher-risk winners       : "
        f"{smoke_report['higher_risk_winners']}"
    )

    print()

    print(
        "Winner distribution:"
    )

    for mode in PREPROCESSING_MODES:

        values = (
            smoke_report[
                "winner_distribution"
            ][
                mode
            ]
        )

        print(
            f"  {mode:<30}"
            f"{values['count']:>3} "
            f"({values['percentage']:>6.2f}%)"
        )

    print()

    print(
        "Runtime damage-gate rejections:"
    )

    rejection_counts = (
        smoke_report[
            "runtime_rejection_reason_counts"
        ]
    )

    if rejection_counts:

        for reason, count in sorted(
            rejection_counts.items(),
            key=lambda item:
                item[1],
            reverse=True,
        ):

            print(
                f"  {reason:<40}"
                f"{count}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Offline fish-detail audit:"
    )

    print(
        f"  Selected winner failures : "
        f"{smoke_report['winner_roi_audit_failures']}"
    )

    roi_failures = (
        smoke_report[
            "winner_roi_failure_reason_counts"
        ]
    )

    if roi_failures:

        for reason, count in sorted(
            roi_failures.items(),
            key=lambda item:
                item[1],
            reverse=True,
        ):

            print(
                f"  {reason:<40}"
                f"{count}"
            )

    print()

    print(
        "Hard protections:"
    )

    print(
        "  Geometry changes         : FORBIDDEN"
    )

    print(
        "  Resolution reduction     : FORBIDDEN"
    )

    print(
        "  Heavy blur/denoise       : FORBIDDEN"
    )

    print(
        "  Synthetic noise          : FORBIDDEN"
    )

    print(
        "  Extreme colour changes   : FORBIDDEN"
    )

    print(
        "  Multi-scale local detail : ENABLED"
    )

    print(
        "  Fish ROI audit           : OFFLINE ONLY"
    )

    print(
        "  Tiny-fish audit          : OFFLINE ONLY"
    )

    print(
        "  Raw fallback             : ENABLED"
    )

    print()

    print(
        "Reports:"
    )

    print(
        f"  "
        f"{REPORTS_ROOT / 'calibration_report.json'}"
    )

    print(
        f"  "
        f"{REPORTS_ROOT / 'sampling_independence_report.json'}"
    )

    print(
        f"  "
        f"{REPORTS_ROOT / 'mode_safety_report.json'}"
    )

    print(
        f"  "
        f"{REPORTS_ROOT / 'smoke_test_report.json'}"
    )

    print(
        f"  "
        f"{REPORTS_ROOT / 'policy_candidate.json'}"
    )

    print()

    print(
        "Comparison grids:"
    )

    print(
        GRIDS_ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT run master_preprocessing.py yet."
    )

    if not calibration_report[
        "calibration_safe"
    ]:

        print(
            "Calibration evidence is not sufficient "
            "for policy approval."
        )

    elif (
        smoke_report[
            "winner_roi_audit_failures"
        ]
        > 0
    ):

        print(
            "At least one selected processed smoke "
            "winner failed the offline fish-detail audit."
        )

        print(
            "Policy v1.4.3 is NOT eligible for approval."
        )

    else:

        print(
            "Calibration achieved meaningful coverage "
            "with zero known fish-detail failures."
        )

        print(
            "Independent smoke winners also passed "
            "the offline fish-detail audit."
        )

        print(
            "Final visual comparison-grid review is "
            "required before freezing the policy."
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("=" * 78)

    print(
        " FISH DETECTION - PREPROCESSING POLICY v1.4.3"
    )

    print("=" * 78)

    print()

    print(
        "Core principle:"
    )

    print(
        "Safety decides whether a candidate may compete."
    )

    print(
        "Benefit decides whether a safe candidate "
        "is worth using."
    )

    print()

    print(
        "Calibration principle:"
    )

    print(
        "Threshold selection is class-neutral."
    )

    print(
        "Fish-positive images are NOT rewarded."
    )

    print(
        "Negative/background images are NOT penalized."
    )

    print()

    print(
        "Ground-truth boxes are used only for "
        "offline fish-detail safety auditing."
    )

    print()

    if not PREPARED_ROOT.exists():

        raise FileNotFoundError(
            "Prepared dataset does not exist."
        )

    # --------------------------------------------------------
    # 1. Load Stage-1 metadata
    # --------------------------------------------------------

    image_index = (
        load_image_index()
    )

    annotation_index = (
        load_annotation_index()
    )

    dataset_info = (
        load_dataset_info()
    )

    train_records = [
        record
        for record
        in image_index
        if record[
            "split"
        ]
        == "train"
    ]

    positive_train = sum(
        1
        for record
        in train_records
        if record[
            "target_count"
        ]
        > 0
    )

    negative_train = (
        len(
            train_records
        )
        - positive_train
    )

    print(
        f"TRAIN images available    : "
        f"{len(train_records):,}"
    )

    print(
        f"Fish-positive             : "
        f"{positive_train:,}"
    )

    print(
        f"Negative/background       : "
        f"{negative_train:,}"
    )

    # --------------------------------------------------------
    # 2. Independent smoke set
    # --------------------------------------------------------

    smoke_samples = (
        choose_smoke_samples(
            train_records,
            annotation_index,
        )
    )

    if (
        len(
            smoke_samples
        )
        != SMOKE_TOTAL_COUNT
    ):

        raise RuntimeError(
            "Incorrect smoke sample count."
        )

    # --------------------------------------------------------
    # 3. Calibration set excluding all smoke deployments
    # --------------------------------------------------------

    calibration_samples = (
        choose_calibration_samples(
            train_records,
            smoke_samples,
            annotation_index,
        )
    )

    if (
        len(
            calibration_samples
        )
        != CALIBRATION_TOTAL_COUNT
    ):

        raise RuntimeError(
            "Incorrect calibration sample count."
        )

    # --------------------------------------------------------
    # 4. Hard sampling-independence preflight
    #
    # This must pass before any expensive image evaluation.
    # --------------------------------------------------------

    sampling_independence_report = (
        sampling_independence_preflight(
            train_records,
            smoke_samples,
            calibration_samples,
        )
    )

    # --------------------------------------------------------
    # 5. Clear previous smoke outputs
    # --------------------------------------------------------

    reset_outputs()

    # --------------------------------------------------------
    # 6. Save sampling evidence and manifests
    # --------------------------------------------------------

    write_json(
        REPORTS_ROOT
        / "sampling_independence_report.json",
        sampling_independence_report,
    )

    write_json(
        REPORTS_ROOT
        / "smoke_sample_manifest.json",

        [
            {
                "sample_number":
                    index,

                "prepared_image":
                    record[
                        "prepared_image"
                    ],

                "deployment":
                    get_deployment(
                        record
                    ),

                "sample_type":
                    (
                        "positive"
                        if record[
                            "target_count"
                        ]
                        > 0
                        else "negative"
                    ),

                "selection_reason":
                    record[
                        "smoke_test_reason"
                    ],

                "target_count":
                    record[
                        "target_count"
                    ],
            }
            for index, record
            in enumerate(
                smoke_samples,
                start=1,
            )
        ],
    )

    write_json(
        REPORTS_ROOT
        / "calibration_sample_manifest.json",

        [
            {
                "prepared_image":
                    record[
                        "prepared_image"
                    ],

                "deployment":
                    get_deployment(
                        record
                    ),

                "sample_type":
                    (
                        "positive"
                        if record[
                            "target_count"
                        ]
                        > 0
                        else "negative"
                    ),

                "target_count":
                    record[
                        "target_count"
                    ],

                "tiny_count":
                    record.get(
                        "_tiny_count",
                        0,
                    ),

                "small_count":
                    record.get(
                        "_small_count",
                        0,
                    ),
            }
            for record
            in calibration_samples
        ],
    )

    # --------------------------------------------------------
    # 7. Evaluate calibration images
    # --------------------------------------------------------

    calibration_rows = (
        evaluate_calibration_set(
            calibration_samples,
            annotation_index,
        )
    )

    # --------------------------------------------------------
    # 8. Mode safety diagnostics
    # --------------------------------------------------------

    mode_report = (
        calibration_mode_report(
            calibration_rows
        )
    )

    write_json(
        REPORTS_ROOT
        / "mode_safety_report.json",
        mode_report,
    )

    # --------------------------------------------------------
    # 9. CLASS-NEUTRAL threshold calibration
    # --------------------------------------------------------

    (
        calibrated_threshold,
        calibration_report,
    ) = calibrate_threshold(
        calibration_rows
    )

    write_json(
        REPORTS_ROOT
        / "calibration_report.json",
        calibration_report,
    )

    # --------------------------------------------------------
    # 10. Independent smoke verification
    # --------------------------------------------------------

    smoke_report = run_smoke_test(
        smoke_samples,
        annotation_index,
        calibrated_threshold,
        calibration_report[
            "calibration_safe"
        ],
    )

    write_json(
        REPORTS_ROOT
        / "smoke_test_report.json",
        smoke_report,
    )

    write_decision_csv(
        smoke_report[
            "decisions"
        ]
    )

    # --------------------------------------------------------
    # 11. Policy candidate
    # --------------------------------------------------------

    policy_candidate = (
        create_policy_candidate(
            calibrated_threshold,
            calibration_report,
            dataset_info,
        )
    )

    write_json(
        REPORTS_ROOT
        / "policy_candidate.json",
        policy_candidate,
    )

    # --------------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------------

    print_summary(
        smoke_report,
        calibration_report,
    )


if __name__ == "__main__":
    main()
