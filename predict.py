

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.config import AUDIO_EXTENSIONS, CHECKPOINTS_DIR, REPORTS_DIR
from src.data.dataloaders import get_device
from src.data.dataset import _standardize
from src.data.features import get_feature_extractor
from src.data.preprocessing import preprocess_waveform
from src.models.lcnn import LCNN, LCNN_IN_CHANNELS

FEATURE_TYPE = "multi"
POOLING = "attentive"
RUN_NAME = "lcnn_multi_attentive_v1"

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / f"{RUN_NAME}_best.pt"
DEFAULT_RESULTS_JSON = REPORTS_DIR / f"{RUN_NAME}_results.json"
DEFAULT_EER_THRESHOLD = 0.5


def load_model(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the primary model first, e.g.:\n"
            f"  python train.py lcnn --feature-type {FEATURE_TYPE} --pooling {POOLING} "
            f"--run-name {RUN_NAME}"
        )
    if device is None:
        device = get_device()

    model = LCNN(in_channels=LCNN_IN_CHANNELS[FEATURE_TYPE], pooling=POOLING)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, device


def get_eer_threshold(results_json: Path | str = DEFAULT_RESULTS_JSON) -> float:
    """Read the test-set EER operating point from a `train.py` results JSON."""
    results_json = Path(results_json)
    if not results_json.exists():
        print(
            f"Warning: {results_json} not found; falling back to "
            f"eer_threshold={DEFAULT_EER_THRESHOLD}. Run evaluation (Phase B) to "
            "produce a calibrated threshold.",
            file=sys.stderr,
        )
        return DEFAULT_EER_THRESHOLD
    report = json.loads(results_json.read_text())
    return float(report["test_metrics"]["eer_threshold"])


def _extract_features(audio_path: Path | str) -> dict[str, torch.Tensor]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format {audio_path.suffix!r}. "
            f"Supported extensions: {sorted(AUDIO_EXTENSIONS)}"
        )

    try:
        waveform = preprocess_waveform(audio_path, mode="eval")
    except Exception as exc:  
        raise RuntimeError(f"Failed to load/process audio file {audio_path}: {exc}") from exc

    features = get_feature_extractor(FEATURE_TYPE)(waveform)
    return {key: _standardize(value) for key, value in features.items()}


@torch.no_grad()
def predict_file(
    audio_path: Path | str,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    eer_threshold: float | None = None,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    results_json: Path | str = DEFAULT_RESULTS_JSON,
) -> dict:
  
    if model is None:
        model, device = load_model(checkpoint_path, device)
    elif device is None:
        device = next(model.parameters()).device

    if eer_threshold is None:
        eer_threshold = get_eer_threshold(results_json)

    features = _extract_features(audio_path)
    inputs = {key: value.unsqueeze(0).to(device) for key, value in features.items()}

    logits = model(inputs)
    p_fake = torch.sigmoid(logits).item()

    if p_fake >= eer_threshold:
        prediction, confidence = "Deepfake", p_fake
    else:
        prediction, confidence = "Genuine", 1.0 - p_fake

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "eer_threshold": round(eer_threshold, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Genuine vs. Deepfake for a single audio file (LCNN, multi+attentive)."
    )
    parser.add_argument("audio_path", type=Path, help="Path to a .wav/.mp3/.flac file.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    args = parser.parse_args()

    result = predict_file(args.audio_path, checkpoint_path=args.checkpoint, results_json=args.results_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
