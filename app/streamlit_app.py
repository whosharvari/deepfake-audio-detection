"""Streamlit demo for the Deepfake Audio Detection project (MARS Open Projects 2026).

Architecture
------------
Single-page app (``st.tabs``) on top of the existing, already-verified
pipeline — no new model/preprocessing code:

- **Predict**: upload a wav/mp3/flac clip, run it through
  ``predict.predict_file`` (Model 1 / LCNN, multi+attentive), and display the
  prediction, calibrated confidence, EER threshold, the fixed 3-second
  model-input waveform, and its Log-Mel/LFCC spectrograms.
- **Model Info**: architecture summary + live parameter count and training
  configuration for the loaded LCNN.
- **Metrics / Cross-Dataset / Ablation**: read-only dashboards over
  ``reports/lcnn_primary_results.json``, ``reports/cross_dataset_results.csv``
  and ``reports/ablation_results.csv``.

Tradeoffs
---------
- CPU-only friendly: ``predict.load_model`` uses ``get_device()`` which falls
  back to CPU automatically (Streamlit Cloud has no GPU/MPS).
- The model and static reports are cached (``st.cache_resource`` /
  ``st.cache_data``) so re-running predictions doesn't reload the checkpoint
  or re-parse JSON/CSV on every interaction.
- Every report/checkpoint load is wrapped so a missing file degrades to an
  ``st.info``/``st.warning`` placeholder instead of crashing the app, in case
  a report or checkpoint is absent on a fresh checkout.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` puts `app/` (not the project root) on
# sys.path[0]; add the project root so `predict` and `src` are importable
# regardless of the working directory (local dev or Streamlit Cloud).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_EXTENSIONS, REPORTS_DIR, SAMPLE_RATE, TARGET_LENGTH_SECONDS  # noqa: E402
from src.data.features import get_feature_extractor  # noqa: E402
from src.data.preprocessing import fix_length, load_audio  # noqa: E402

import predict  # noqa: E402

st.set_page_config(page_title="Deepfake Audio Detection", layout="wide")


@st.cache_resource(show_spinner="Loading LCNN model...")
def _load_predictor():
    """Returns ``(model, device, eer_threshold, error)``; `error` is ``None`` on success."""
    try:
        model, device = predict.load_model()
    except FileNotFoundError as exc:
        return None, None, predict.DEFAULT_EER_THRESHOLD, str(exc)
    eer_threshold = predict.get_eer_threshold()
    return model, device, eer_threshold, None


@st.cache_data(show_spinner=False)
def _load_json_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def _load_csv_report(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _plot_waveform(waveform) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    samples = waveform.squeeze().numpy()
    t = [i / SAMPLE_RATE for i in range(len(samples))]
    ax.plot(t, samples, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (fixed 3s model-input window)")
    fig.tight_layout()
    return fig


def _plot_spectrogram(spec, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(spec.squeeze().numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def render_predict_tab() -> None:
    st.header("Upload an audio clip")
    uploaded = st.file_uploader("WAV / MP3 / FLAC", type=["wav", "mp3", "flac"])
    if uploaded is None:
        st.info("Upload a .wav, .mp3, or .flac file to get a prediction.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        st.error(f"Unsupported file extension {suffix!r}. Supported: {sorted(AUDIO_EXTENSIONS)}")
        return

    st.audio(uploaded)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        full_waveform = load_audio(tmp_path)
        original_seconds = full_waveform.shape[-1] / SAMPLE_RATE
        windowed = fix_length(full_waveform, mode="eval")

        model, device, eer_threshold, load_error = _load_predictor()
        if load_error is not None:
            st.warning(f"Model checkpoint not available yet: {load_error}")
        else:
            result = predict.predict_file(tmp_path, model=model, device=device, eer_threshold=eer_threshold)
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result["prediction"])
            col2.metric("Confidence", f"{result['confidence']:.2%}")
            col3.metric("EER threshold", f"{result['eer_threshold']:.4f}")

        st.caption(
            f"Original clip duration: {original_seconds:.2f}s -> fixed {TARGET_LENGTH_SECONDS:.0f}s "
            "model-input window (wrap-padded if shorter, center-cropped if longer). "
            "Every clip is normalized to this same length, so the model cannot use "
            "duration as a shortcut."
        )

        st.subheader("Waveform")
        st.pyplot(_plot_waveform(windowed))

        feats = get_feature_extractor("multi")(windowed)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Log-Mel Spectrogram")
            st.pyplot(_plot_spectrogram(feats["logmel"], "Log-Mel (80 bins)", "Mel bin"))
        with col2:
            st.subheader("LFCC")
            st.pyplot(_plot_spectrogram(feats["lfcc"], "LFCC (40 coefficients)", "Coefficient"))

    except Exception as exc:  # noqa: BLE001 - show a friendly error instead of crashing the app
        st.error(f"Failed to process audio file: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def render_model_info_tab() -> None:
    st.header("Model: LCNN (Model 1, primary)")
    st.markdown(
        """
        - **Input**: 2-channel Log-Mel(80) + LFCC(40, bilinearly resized to 80) spectrogram pair,
          fixed 3-second / 48,000-sample window
        - **Stem**: 5x5 conv -> Max-Feature-Map (MFM) -> BatchNorm -> 2x2 max-pool
        - **Backbone**: 3 residual blocks (32->64->128->128 channels, stride 2 each), each with
          two MFM-activated conv stages
        - **Pooling**: Attentive Statistics Pooling (learned per-frame attention over mean/std)
        - **Head**: Linear(1280->256) -> ReLU -> Dropout -> Linear(256->1)
        - **Output**: raw logit; ``sigmoid(logit) = P(deepfake)``
        """
    )

    model, device, eer_threshold, load_error = _load_predictor()
    if load_error is not None:
        st.warning(f"Checkpoint not available yet: {load_error}")
    else:
        n_params = sum(p.numel() for p in model.parameters())
        col1, col2, col3 = st.columns(3)
        col1.metric("Parameters", f"{n_params:,}")
        col2.metric("Inference device", str(device))
        col3.metric("EER threshold (test)", f"{eer_threshold:.4f}")

        report = _load_json_report(predict.DEFAULT_RESULTS_JSON)
        if report is not None and "args" in report:
            st.subheader("Training configuration")
            st.json(report["args"])


def render_metrics_tab() -> None:
    st.header("Primary model: validation & test metrics")
    report = _load_json_report(REPORTS_DIR / "lcnn_primary_results.json")
    if report is None:
        st.info("reports/lcnn_primary_results.json not found - run evaluate_primary.py.")
        return

    for split_key, title in (("val_metrics", "Validation"), ("test_metrics", "Test")):
        if split_key not in report:
            continue
        st.subheader(title)
        metrics = report[split_key]
        cols = st.columns(6)
        for col, key in zip(cols, ("accuracy", "precision", "recall", "f1", "roc_auc", "eer")):
            if key in metrics:
                col.metric(key.upper(), f"{metrics[key]:.4f}")

        cm = metrics.get("confusion_matrix")
        if cm:
            st.write("Confusion matrix (rows = true, cols = predicted; order = [real, fake]):")
            st.dataframe(pd.DataFrame(cm, index=["real", "fake"], columns=["pred_real", "pred_fake"]))

        pca = metrics.get("per_class_accuracy")
        if pca:
            st.write("Per-class accuracy:", pca)

    history = report.get("history")
    if history:
        st.subheader("Training curves")
        hist_df = pd.DataFrame(history)
        hist_df.index = hist_df.index + 1
        hist_df.index.name = "epoch"
        st.line_chart(hist_df[["train_loss", "val_loss"]])
        extra_cols = [c for c in ("val_eer", "val_roc_auc") if c in hist_df.columns]
        if extra_cols:
            st.line_chart(hist_df[extra_cols])


def render_cross_dataset_tab() -> None:
    st.header("Cross-dataset generalization (Experiments A/B/C)")
    df = _load_csv_report(REPORTS_DIR / "cross_dataset_results.csv")
    if df is None:
        st.info("reports/cross_dataset_results.csv not found - run evaluate_cross_dataset.py.")
        return
    st.dataframe(df)
    metric_cols = [c for c in ("accuracy", "eer", "roc_auc") if c in df.columns]
    if "source_dataset" in df.columns and metric_cols:
        st.bar_chart(df.set_index("source_dataset")[metric_cols])


def render_ablation_tab() -> None:
    st.header("Ablation study")
    df = _load_csv_report(REPORTS_DIR / "ablation_results.csv")
    if df is None:
        st.info("reports/ablation_results.csv not found - run build_ablation_report.py.")
        return
    st.dataframe(df)
    label_col = "run_name" if "run_name" in df.columns else df.columns[0]
    metric_cols = [c for c in ("accuracy", "f1", "roc_auc", "eer") if c in df.columns]
    if metric_cols:
        st.bar_chart(df.set_index(label_col)[metric_cols])


def main() -> None:
    st.title("Deepfake Audio Detection")
    st.caption("MARS Open Projects 2026 - Problem Statement 2 (Genuine vs. AI-generated speech)")

    tabs = st.tabs(["Predict", "Model Info", "Metrics", "Cross-Dataset", "Ablation"])
    with tabs[0]:
        render_predict_tab()
    with tabs[1]:
        render_model_info_tab()
    with tabs[2]:
        render_metrics_tab()
    with tabs[3]:
        render_cross_dataset_tab()
    with tabs[4]:
        render_ablation_tab()


if __name__ == "__main__":
    main()
