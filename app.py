import os
import base64
import textwrap
import streamlit as st
from PIL import Image

from model_utils import (
    load_model_artifact,
    predict_damage,
    CATEGORY_DISPLAY_NAMES,
    CATEGORY_ICONS
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Helper function to prevent Markdown from parsing indented HTML as code blocks
def render_html(html_str: str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Style Injection
# -----------------------------------------------------------------------------
def get_b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def inject_styles():
    bg_css = ""
    bg_path = os.path.join("assets", "car_bg.jpg")
    if os.path.exists(bg_path):
        try:
            b64_str = get_b64_file(bg_path)
            bg_css = f"""
            .stApp {{
                background: linear-gradient(rgba(11, 15, 25, 0.90), rgba(11, 15, 25, 0.94)), url("data:image/jpeg;base64,{b64_str}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            """
        except Exception:
            pass

    css_path = os.path.join("assets", "style.css")
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    st.markdown(f"<style>{bg_css}\n{css_content}</style>", unsafe_allow_html=True)


inject_styles()


# -----------------------------------------------------------------------------
# Model Caching Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model():
    return load_model_artifact("car_damage_model.pkl")


model_loaded = True
model_error_msg = ""
try:
    model_artifact = load_cached_model()
except Exception as e:
    model_loaded = False
    model_error_msg = str(e)


# -----------------------------------------------------------------------------
# Top Hero Banner
# -----------------------------------------------------------------------------
render_html("""
<div class="hero-header-card">
    <div class="hero-status-pill">SYSTEM ONLINE • CLASSIFIER READY</div>
    <h1 class="hero-main-title">Car Damage Assessment AI</h1>
    <p class="hero-main-sub">Automated Multi-Label Vehicle Damage Detection & Extent Classification System</p>
    <div class="hero-tag-row">
        <span class="hero-tag-pill">OpenCV Vision Stream</span>
        <span class="hero-tag-pill">HOG + LBP Features</span>
        <span class="hero-tag-pill">StandardScaler</span>
        <span class="hero-tag-pill">PCA Dimensionality Reduction</span>
        <span class="hero-tag-pill">One-vs-Rest LinearSVC</span>
        <span class="hero-tag-pill">CarDD Dataset</span>
    </div>
</div>
""")


if not model_loaded:
    st.error(
        f"⚠️ **Model Artifact Error**: Unable to load `car_damage_model.pkl`.\n\n"
        f"Details: `{model_error_msg}`\n\n"
        f"Please verify that `car_damage_model.pkl` is located in the application directory."
    )
    st.stop()


# -----------------------------------------------------------------------------
# KPI Summary Grid
# -----------------------------------------------------------------------------
render_html("""
<div class="kpi-summary-grid">
    <div class="kpi-card-item">
        <div class="kpi-card-lbl">ML Architecture</div>
        <div class="kpi-card-val">HOG + LBP + SVM</div>
        <div class="kpi-card-sub">Handcrafted Classical CV</div>
    </div>
    <div class="kpi-card-item">
        <div class="kpi-card-lbl">Feature Dimensions</div>
        <div class="kpi-card-val">8,110 ➔ 1,335</div>
        <div class="kpi-card-sub">PCA 95% Retained Variance</div>
    </div>
    <div class="kpi-card-item">
        <div class="kpi-card-lbl">Supported Classes</div>
        <div class="kpi-card-val">6 Categories</div>
        <div class="kpi-card-sub">Multi-Label Prediction</div>
    </div>
    <div class="kpi-card-item">
        <div class="kpi-card-lbl">Model Micro F1</div>
        <div class="kpi-card-val">0.515 Score</div>
        <div class="kpi-card-sub">Validation Set Performance</div>
    </div>
</div>
""")


# -----------------------------------------------------------------------------
# Main Application Columns
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    st.markdown("### 📥 Vehicle Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a car image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a photograph of the damaged vehicle."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        if st.session_state.get("current_file_name") != file_name:
            st.session_state.current_file_name = file_name
            st.session_state.current_file_bytes = file_bytes
            st.session_state.assessment_done = False

        try:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption=f"Uploaded Image: {file_name}", use_container_width=True)
            st.caption(f"Resolution: {pil_image.width}×{pil_image.height} px • File Size: {len(file_bytes)/1024:.1f} KB")
        except Exception:
            st.error("Unable to load image preview.")

        analyze_button = st.button("⚡ ASSESS VEHICLE DAMAGE", type="primary", use_container_width=True)
    else:
        file_bytes = None
        analyze_button = False
        st.session_state.assessment_done = False
        st.info("💡 **Instructions:** Upload a photograph of the damaged vehicle section above and click **⚡ ASSESS VEHICLE DAMAGE** to run AI classification.")

    render_html("""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-top: 16px;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;">OpenCV Preprocessing Stream</div>
        <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
            • BGR to Grayscale conversion<br>
            • Resized to exact 128×128 pixel matrix<br>
            • Gaussian Blur (3×3 kernel) smoothing<br>
            • Pixel normalization to [0.0, 1.0] float32
        </div>
    </div>
    """)


with col_right:
    st.markdown("### 📊 Assessment & Damage Analysis")

    # Perform inference ONLY on button click
    if uploaded_file is not None and analyze_button:
        with st.spinner("Extracting HOG (8100-D) + LBP (10-D) ➔ PCA ➔ Calibrated LinearSVC..."):
            try:
                detected_cats, probabilities, decision_scores, auto_severity = predict_damage(file_bytes, model_artifact)
                st.session_state.detected_cats = detected_cats
                st.session_state.probabilities = probabilities
                st.session_state.decision_scores = decision_scores
                st.session_state.auto_severity = auto_severity
                st.session_state.assessment_done = True
            except Exception as ex:
                st.error(f"Assessment error: {ex}")
                st.session_state.assessment_done = False

    # Render results ONLY after button has been clicked
    if st.session_state.get("assessment_done", False) and uploaded_file is not None:
        detected_cats = st.session_state.detected_cats
        probabilities = st.session_state.probabilities
        auto_severity = st.session_state.auto_severity

        # ---------------------------------------------------------------------
        # 1. Detected Damage Section
        # ---------------------------------------------------------------------
        st.markdown("#### 🛠️ Multi-Label Damage Detection")

        if len(detected_cats) > 0:
            st.write("The calibrated multi-label classifier identified the following damage categories:")

            cards_html = '<div class="damage-card-grid">'
            for cat in detected_cats:
                display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat.title())
                icon = CATEGORY_ICONS.get(cat, "🔍")
                prob = probabilities.get(cat, 0.0)
                cards_html += f'<div class="damage-card-box"><div class="damage-card-name">{icon} {display_name}</div><div class="damage-card-prob">Confidence: {prob:.1%}</div></div>'
            cards_html += '</div>'

            render_html(cards_html)

            # Display Automated Damage Severity Extent Badge
            extent_label_map = {
                "severe": ("🔴 SEVERE DAMAGE EXTENT", "extent-badge-severe", "High-Impact Collision / Structural Surface Damage"),
                "moderate": ("🟡 MODERATE DAMAGE EXTENT", "extent-badge-moderate", "Medium Impact Panel Deformation"),
                "minor": ("🟢 MINOR DAMAGE EXTENT", "extent-badge-minor", "Superficial Scratch or Small Dent")
            }
            ext_title, ext_cls, ext_desc = extent_label_map.get(auto_severity, ("MODERATE", "extent-badge-moderate", ""))

            render_html(f"""
            <div class="extent-badge-container {ext_cls}">
                <span>{ext_title}</span> &nbsp;•&nbsp; <span style="font-weight: 400; font-size: 0.85rem;">{ext_desc}</span>
            </div>
            """)

            st.markdown("##### Detailed Confidence Breakdown Across All Categories:")
            for cat, name in CATEGORY_DISPLAY_NAMES.items():
                prob = probabilities.get(cat, 0.0)
                icon = CATEGORY_ICONS.get(cat, "")
                st.progress(prob, text=f"{icon} **{name}**: {prob:.1%}")

        else:
            render_html("""
            <div style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.35); border-radius: 12px; padding: 20px; margin: 10px 0;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #4ade80; margin-bottom: 6px;">ℹ️ No Clear Damage Detected</div>
                <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
                    The trained model did not identify any supported damage category in this image.
                    <br><br>
                    <em>Note: This does not guarantee that the vehicle is damage-free. The system evaluates six specific categories: Dent, Scratch, Crack, Glass Shatter, Lamp Broken, and Tire Flat.</em>
                </div>
            </div>
            """)

    elif uploaded_file is not None:
        render_html("""
        <div class="waiting-prompt-card">
            <div class="waiting-prompt-title">📷 Vehicle Image Ready for Assessment</div>
            <div>Click the <strong>⚡ ASSESS VEHICLE DAMAGE</strong> button below the image preview on the left to run AI multi-label damage classification.</div>
        </div>
        """)
    else:
        render_html("""
        <div class="waiting-prompt-card">
            <div class="waiting-prompt-title">👈 Upload Image to Begin</div>
            <div>Select a car damage photograph on the left to start the assessment.</div>
        </div>
        """)


# -----------------------------------------------------------------------------
# Bottom Section Tabs
# -----------------------------------------------------------------------------
st.markdown("<br><hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

tab_method, tab_pipe, tab_info = st.tabs([
    "📐 Severity Methodology & Disclaimers",
    "⚙️ Computer Vision Workflow",
    "📖 Model Evaluation & Dataset Info"
])

with tab_method:
    st.markdown("""
    #### Visible Damage Severity Methodology
    Visible damage severity bands (**Minor**, **Moderate**, **Severe**) in this project are derived from the distribution of annotated damage-area ratios in the CarDD training dataset:
    - **Minor Extent:** Damage area ratio ≤ 33rd percentile of CarDD training set.
    - **Moderate Extent:** Damage area ratio between 33rd and 67th percentiles.
    - **Severe Extent:** Damage area ratio > 67th percentile.

    ⚠️ **Technical Limitation Note:** The current deployed classifier predicts damage categories from handcrafted HOG+LBP features but *does not perform pixel-level segmentation or direct surface area measurement on newly uploaded images*.
    """)

with tab_pipe:
    st.markdown("#### End-to-End Inference Workflow")
    render_html("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 14px;">
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">1</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">Uploaded Image</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">2</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">OpenCV Preprocessing</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">3</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">HOG + LBP Descriptors</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">4</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">StandardScaler</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">5</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">PCA (95% Variance)</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">6</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">OneVsRest LinearSVC</div>
        </div>
        <div style="background: rgba(15,23,42,0.9); border: 1px solid #1e293b; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="background: #0284c7; color: white; width: 24px; height: 24px; border-radius: 50%; margin: 0 auto 6px auto; font-weight: 800; font-size: 0.78rem; line-height: 24px;">7</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">Damage Categories</div>
        </div>
    </div>
    """)

with tab_info:
    st.markdown("""
    #### Model Performance & Dataset Information
    - **Dataset:** CarDD (Car Damage Dataset) comprising 4,000 high-resolution images with 9,000+ COCO-style damage instances across 6 major categories.
    - **Categories:** Dent, Scratch, Crack, Glass Shatter, Lamp Broken, Tire Flat.

    | Metric | Validation Set | Test Set |
    | :--- | :---: | :---: |
    | **Micro F1-Score** | **0.515** | **0.488** |
    | **Hamming Loss** | 0.224 | 0.236 |
    | **Micro Precision** | 0.395 | 0.380 |
    | **Micro Recall** | 0.741 | 0.682 |
    | **Macro F1-Score** | 0.478 | 0.443 |
    """)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
render_html("""
<div class="footer-info-box">
    <div><strong>Car Damage Assessment AI</strong></div>
    <div>Built with Python, Streamlit, OpenCV, scikit-image, and scikit-learn</div>
    <div style="margin-top: 4px;">CarDD Dataset • Classical Computer Vision & Machine Learning</div>
</div>
""")
