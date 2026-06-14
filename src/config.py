

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
MANIFEST_DIR = PROJECT_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

DATASET_VERSIONS = {
    "for-2sec": DATASET_ROOT / "for-2sec" / "for-2seconds",
    "for-norm": DATASET_ROOT / "for-norm" / "for-norm",
    "for-original": DATASET_ROOT / "for-original" / "for-original",
    "for-rerec": DATASET_ROOT / "for-rerec" / "for-rerecorded",
}

SPLIT_DIR_TO_LABEL = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

LABEL_TO_INT = {"real": 0, "fake": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

INVALID_DURATION = -1.0
INVALID_SAMPLE_RATE = -1

SAMPLE_RATE = 16000

TARGET_LENGTH_SECONDS = 3.0
TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * TARGET_LENGTH_SECONDS) 
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

N_MELS = 80

N_LFCC = 40
N_LFCC_FILTERS = 128

RANDOM_SEED = 42

PRIMARY_SOURCE_DATASETS = ["for-norm"]
