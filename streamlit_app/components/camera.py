# ── Streamlit Camera Component ─────────────────────────────
# Wraps st.camera_input() with preprocessing for YOLO inference
import streamlit as st
from PIL import Image
import numpy as np
import io


def camera_input_section() -> Image.Image | None:
    """
    Renders the camera capture widget with instructions.
    Returns a PIL Image if a photo was taken, else None.
    """
    st.markdown("### 📸 Scan Your Ingredients")
    st.caption(
        "Point your camera at ingredients on a counter or in your fridge. "
        "The AI will detect them in real time."
    )

    col_cam, col_upload = st.columns([2, 1])

    with col_cam:
        img_file = st.camera_input(
            label="Capture ingredients",
            label_visibility="collapsed",
            key="camera_capture",
        )

    with col_upload:
        st.markdown("**Or upload an image:**")
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="upload_fallback",
        )

    # Camera takes priority
    source = img_file or uploaded

    if source is not None:
        image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
        return image

    return None


def display_annotated_image(annotated_img: np.ndarray, label: str = "Detected Ingredients"):
    """Display a YOLO-annotated numpy BGR image in Streamlit."""
    # Convert BGR → RGB for Streamlit
    rgb = annotated_img[:, :, ::-1] if annotated_img.ndim == 3 else annotated_img
    st.image(rgb, caption=label, use_column_width=True)
