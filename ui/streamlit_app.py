"""Clinical screening & Grad-CAM++ dashboard connected to the FastAPI backend."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
SAMPLES_DIR = Path("data/samples/diabetic_retinopathy")
CLASS_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

st.set_page_config(
    page_title="DR Clinical Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1 {
            color: #023E8A;
            border-bottom: 2px solid #ADE8F4;
            padding-bottom: 0.45rem;
            margin-bottom: 1rem;
        }
        h2, h3 { color: #023E8A; }
        [data-testid="stSidebar"] {
            border-right: 1px solid #90E0EF;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #90E0EF;
            border-top: 3px solid #0096C7;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            box-shadow: 0 2px 6px rgba(3, 4, 94, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #0077B6;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #03045E;
        }
        .triage-card {
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin: 0.75rem 0 1rem 0;
            border-left: 5px solid;
        }
        .triage-review {
            background-color: #FFF1F2;
            border-left-color: #C1121F;
            color: #660708;
        }
        .triage-routine {
            background-color: #CAF0F8;
            border-left-color: #0077B6;
            color: #03045E;
        }
        .metric-caption {
            font-size: 0.82rem;
            color: #0096C7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health(base_url: str) -> tuple[bool, dict | str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=4)
        if resp.ok:
            return True, resp.json()
        return False, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def call_inference(
    base_url: str,
    image_bytes: bytes,
    filename: str,
    include_explain: bool = True,
) -> tuple[dict | None, str | None]:
    endpoint = "/api/v1/dr/explain" if include_explain else "/api/v1/dr/predict"
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, image_bytes, "image/png")},
            timeout=60,
        )
        if resp.ok:
            return resp.json(), None

        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"API Error ({resp.status_code}): {detail}"
    except requests.RequestException as exc:
        return None, f"Connection error: {exc}"


def fetch_history(base_url: str, limit: int = 50) -> tuple[list[dict], str | None]:
    url = f"{base_url.rstrip('/')}/api/v1/dr/history"
    try:
        resp = requests.get(url, params={"limit": limit}, timeout=10)
        if not resp.ok:
            return [], f"HTTP {resp.status_code}: {resp.text}"
        payload = resp.json()
        if isinstance(payload, list):
            return payload, None
        for key in ("items", "records", "history", "data"):
            if isinstance(payload.get(key), list):
                return payload[key], None
        return [], "Unexpected history response format."
    except requests.RequestException as exc:
        return [], f"Connection error: {exc}"


def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def plot_class_probabilities(
    probabilities: dict[str, float], predicted: str
) -> plt.Figure:
    labels = [cls for cls in CLASS_ORDER if cls in probabilities] or list(probabilities)
    values = [probabilities[cls] * 100.0 for cls in labels]
    colors = ["#0077B6" if cls == predicted else "#90E0EF" for cls in labels]
    edge_colors = ["#03045E" if cls == predicted else "#0096C7" for cls in labels]

    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7FCFE")

    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=colors[::-1],
        edgecolor=edge_colors[::-1],
        linewidth=0.8,
    )
    ax.set_xlim(0, 105)
    ax.set_xlabel(
        "Calibrated Probability (%)", fontsize=9, color="#023E8A", fontweight="bold"
    )
    ax.tick_params(colors="#03045E", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#0096C7")
    ax.spines["bottom"].set_color("#0096C7")
    ax.grid(axis="x", linestyle=":", alpha=0.55, color="#48CAE4")

    for bar, val in zip(bars, values[::-1], strict=True):
        ax.text(
            min(val + 1.5, 98.0),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=8.5,
            color="#03045E",
            fontweight="bold" if val >= 50 else "normal",
        )

    fig.tight_layout()
    return fig


def render_triage_banner(triage: dict) -> None:
    status = triage.get("triage_status", "UNKNOWN")
    reasons = triage.get("review_reasons", [])
    is_review = status == "NEEDS_SPECIALIST_REVIEW"

    css_class = "triage-review" if is_review else "triage-routine"
    title = "Specialist Review Recommended" if is_review else "Routine Screening"
    reasons_html = ""
    if reasons:
        items = "".join(f"<li>{reason}</li>" for reason in reasons)
        reasons_html = f"<ul style='margin:0.4rem 0 0 1.2rem;'>{items}</ul>"

    st.markdown(
        f"""
        <div class="triage-card {css_class}">
            <strong>{title}</strong> <code>({status})</code>
            {reasons_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_local_samples() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        p
        for p in SAMPLES_DIR.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Service Configuration")
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL)
    include_explain = st.toggle("Generate Grad-CAM++ Overlay", value=True)

    is_healthy, health_info = check_api_health(api_url)
    if is_healthy:
        st.success("API Online")
    else:
        st.error(f"API Offline ({health_info})")

    st.divider()
    st.caption(
        "Research & decision-support interface only. Not a certified medical device "
        "and must not replace ophthalmologist evaluation."
    )


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.title("Diabetic Retinopathy Screening")

tab_screen, tab_history = st.tabs(["Screening & Explainability", "Audit History"])

with tab_screen:
    uploaded = st.file_uploader(
        "Upload retinal fundus image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
    )

    sample_files = load_local_samples()
    selected_sample: Path | None = None
    if sample_files:
        sample_labels = ["— None —"] + [p.name for p in sample_files]
        choice = st.selectbox("Or load a local diagnostic sample", sample_labels)
        if choice != "— None —":
            selected_sample = next(p for p in sample_files if p.name == choice)

    image_bytes: bytes | None = None
    filename = "fundus.png"

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name
    elif selected_sample is not None:
        image_bytes = selected_sample.read_bytes()
        filename = selected_sample.name

    if image_bytes is not None:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        run_clicked = st.button(
            "Run Screening", type="primary", use_container_width=True
        )
        if run_clicked:
            with st.spinner("Running calibrated inference..."):
                result, error = call_inference(
                    base_url=api_url,
                    image_bytes=image_bytes,
                    filename=filename,
                    include_explain=include_explain,
                )

            if error:
                st.error(error)
            elif result is not None:
                st.session_state["last_result"] = result
                st.session_state["last_image"] = original_image

        if "last_result" in st.session_state and "last_image" in st.session_state:
            res = st.session_state["last_result"]
            img_orig = st.session_state["last_image"]

            render_triage_banner(res.get("triage", {}))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Grade", res.get("predicted_class", "—"))
            m2.metric(
                "Calibrated Confidence", f"{res.get('confidence', 0.0) * 100:.2f}%"
            )
            m3.metric("Predictive Entropy", f"{res.get('entropy', 0.0):.4f}")
            m4.metric(
                "Latency / Temp",
                f"{res.get('inference_time_ms', 0):.0f} ms",
                delta=f"T = {res.get('calibration_temperature', 1.0):.4f}",
                delta_color="off",
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Original Fundus Image**")
                st.image(img_orig, use_container_width=True)

            with col_right:
                overlay_b64 = res.get("gradcam_overlay_base64")
                if overlay_b64:
                    cam_method = res.get("cam_method", "gradcam++")
                    target_layer = res.get("target_layer", "layer4")
                    st.markdown(f"**{cam_method.upper()} Saliency (`{target_layer}`)**")
                    overlay_img = decode_base64_image(overlay_b64)
                    st.image(overlay_img, use_container_width=True)
                else:
                    st.info(
                        "Enable 'Generate Grad-CAM++ Overlay' in the sidebar to inspect saliency maps."
                    )

            st.markdown("**Calibrated Class Probabilities**")
            probs = res.get("probabilities", {})
            if probs:
                fig = plot_class_probabilities(probs, res.get("predicted_class", ""))
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

            with st.expander("Raw API Response"):
                preview = {
                    k: ("<base64_png_omitted>" if k == "gradcam_overlay_base64" else v)
                    for k, v in res.items()
                }
                st.json(preview)
    else:
        st.info(
            "Upload a retinal fundus image or choose a local diagnostic sample to begin."
        )


with tab_history:
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        limit = st.number_input(
            "Records limit", min_value=10, max_value=200, value=50, step=10
        )
    with col_ctrl2:
        st.write("")
        refresh = st.button("Refresh History")

    records, hist_err = fetch_history(api_url, limit=int(limit))
    if hist_err:
        st.warning(f"Could not load audit history: {hist_err}")
    elif not records:
        st.info("No inference records logged yet.")
    else:
        df = pd.DataFrame(records)

        if "triage_status" in df.columns:
            statuses = ["All"] + sorted(df["triage_status"].dropna().unique().tolist())
            chosen_status = st.selectbox("Filter by triage status", statuses)
            if chosen_status != "All":
                df = df[df["triage_status"] == chosen_status]

        st.dataframe(df, use_container_width=True, hide_index=True)
