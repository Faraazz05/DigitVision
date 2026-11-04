import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from torchvision import datasets

from model import DigitCNN

# Constants & Paths
MEAN, STD = 0.1307, 0.3081
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "digit_cnn.pt"
DATA_DIR = BASE_DIR / "data"

# Page configuration
st.set_page_config(
    page_title="DigitVision - Interactive CNN & Explainable AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #161a30 0%, #0b0d18 70%);
        color: #f1f3f9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .tech-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 46, 0.85);
        backdrop-filter: blur(10px);
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #28304a;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .card {
        background: rgba(22, 27, 46, 0.7);
        border-radius: 12px;
        border: 1px solid #28304a;
        padding: 18px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained CNN model weights."""
    if not MODEL_PATH.exists():
        st.error(f"Model file '{MODEL_PATH.name}' not found. Please run 'python train.py' first.")
        st.stop()
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_data
def get_mnist_benchmark_samples():
    """Cache benchmark indices from MNIST test dataset for instant testing."""
    if not (DATA_DIR / "MNIST").exists():
        return {}
    try:
        ds = datasets.MNIST(str(DATA_DIR), train=False, download=False)
        samples_by_digit = {i: [] for i in range(10)}
        for idx in range(min(len(ds), 1200)):
            _, label = ds[idx]
            if len(samples_by_digit[label]) < 4:
                samples_by_digit[label].append(idx)
        return samples_by_digit
    except Exception:
        return {}


def preprocess_canvas_or_image(img_data):
    """Convert canvas or raw image array into a centered 28x28 grayscale image."""
    if img_data is None:
        return None
    if isinstance(img_data, np.ndarray):
        if img_data.size == 0:
            return None
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            gray = Image.fromarray(img_data[:, :, :3].astype("uint8")).convert("L")
        else:
            gray = Image.fromarray(img_data.astype("uint8")).convert("L")
    elif isinstance(img_data, Image.Image):
        gray = img_data.convert("L")
    else:
        return None

    arr = np.array(gray)
    if arr.max() < 30:
        return None

    # Crop to bounding box of pixels > 30
    ys, xs = np.where(arr > 30)
    if len(xs) == 0 or len(ys) == 0:
        return None

    gray = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = gray.size
    if w == 0 or h == 0:
        return None

    # Resize so largest dimension fits 20x20, maintaining aspect ratio
    scale = 20 / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    gray = gray.resize((new_w, new_h), Image.LANCZOS)

    # Paste centered into 28x28 black canvas (standard MNIST format)
    out = Image.new("L", (28, 28), 0)
    out.paste(gray, ((28 - new_w) // 2, (28 - new_h) // 2))
    return out


def apply_colormap(norm_arr, cmap="inferno"):
    """Apply a smooth, high-fidelity colormap to a 2D [0, 1] normalized array."""
    x = np.clip(norm_arr, 0.0, 1.0)
    if cmap == "inferno":
        r = np.clip(1.6 * x - 0.2, 0, 1) * np.clip(2.0 * x, 0, 1)
        g = np.clip(1.8 * x - 0.7, 0, 1)
        b = np.clip(0.6 * np.sin(x * np.pi) + (x > 0.8) * 0.4, 0, 1)
    elif cmap == "jet":
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    elif cmap == "viridis":
        r = np.clip(0.28 + 0.7 * (x**1.5) - 0.1 * np.sin(x * np.pi), 0, 1)
        g = np.clip(0.1 + 0.9 * x, 0, 1)
        b = np.clip(0.48 + 0.5 * np.cos(x * np.pi * 0.8), 0, 1)
    else:  # plasma
        r = np.clip(0.5 + 0.5 * np.sin(np.pi * (x - 0.5)), 0, 1)
        g = np.clip(np.sin(np.pi * x * 0.8), 0, 1)
        b = np.clip(np.cos(np.pi * x * 0.5), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def compute_xai_map(model, x, pred_class, method="gradcam"):
    """Compute attribution heatmap using Grad-CAM, Saliency, or Input x Gradient."""
    x = x.clone().detach().requires_grad_(True)
    activations, gradients = [], []

    def f_hook(m, inp, outp):
        activations.append(outp)

    def b_hook(m, gin, gout):
        gradients.append(gout[0])

    h1 = model.conv2.register_forward_hook(f_hook)
    h2 = model.conv2.register_full_backward_hook(b_hook)

    logits = model(x)
    logits[0, pred_class].backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if method == "gradcam" and activations and gradients:
        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
        m = cam[0, 0].detach().cpu().numpy()
    elif method == "input_grad":
        m = (x * x.grad).abs()[0, 0].detach().cpu().numpy()
    else:  # vanilla saliency
        m = x.grad.abs()[0, 0].detach().cpu().numpy()

    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m_norm


def run_fgsm_attack(model, x, pred_class, epsilon):
    """Perform Fast Gradient Sign Method (FGSM) adversarial attack."""
    if epsilon == 0.0:
        return x, None, pred_class, 1.0

    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, torch.tensor([pred_class]))
    loss.backward()

    perturbation = x_adv.grad.sign()
    perturbed_x = x + epsilon * perturbation
    # Re-normalize to valid MNIST normalized range
    min_norm = (0.0 - MEAN) / STD
    max_norm = (1.0 - MEAN) / STD
    perturbed_x = torch.clamp(perturbed_x, min_norm, max_norm)

    with torch.no_grad():
        adv_logits = model(perturbed_x)
        adv_probs = torch.softmax(adv_logits, dim=1)[0].cpu().numpy()
        adv_pred = adv_logits.argmax(1).item()

    return perturbed_x, perturbation[0, 0].cpu().numpy(), adv_pred, adv_probs[adv_pred]


def extract_layer_feature_maps(model, x):
    """Extract activation maps for Conv1 and Conv2 layers."""
    with torch.no_grad():
        c1 = F.relu(model.conv1(x))[0].cpu().numpy()
        p1 = F.max_pool2d(F.relu(model.conv1(x)), 2)
        c2 = F.relu(model.conv2(p1))[0].cpu().numpy()
    return c1, c2


def build_feature_grid_image(maps, num_filters=8, cols=4):
    """Combine multiple filter activation maps into a single composite image."""
    selected = maps[:num_filters]
    h, w = selected.shape[1], selected.shape[2]
    rows = (len(selected) + cols - 1) // cols
    padding = 2
    grid = np.zeros((rows * (h + padding), cols * (w + padding)), dtype=np.uint8)

    for idx, f in enumerate(selected):
        r = idx // cols
        c = idx % cols
        f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
        f_img = (f_norm * 255).astype(np.uint8)
        y = r * (h + padding)
        x = c * (w + padding)
        grid[y : y + h, x : x + w] = f_img

    return Image.fromarray(grid).resize((cols * 56, rows * 56), Image.NEAREST)


def generate_diagnostic_card(orig_img, heatmap_img, pred, conf, entropy, top3):
    """Generate an exportable summary report card as PNG bytes."""
    card_w, card_h = 620, 340
    card = Image.new("RGB", (card_w, card_h), (18, 22, 38))
    draw = ImageDraw.Draw(card)

    # Header bar
    draw.rectangle([(0, 0), (card_w, 42)], fill=(28, 34, 60))
    draw.text((16, 12), "DigitVision AI - Diagnostic Report", fill=(167, 139, 250))

    # Paste input and heatmap
    draw.text((30, 52), "Processed Input (28x28)", fill=(148, 163, 184))
    card.paste(orig_img.resize((120, 120), Image.NEAREST), (30, 75))

    draw.text((175, 52), "Attribution Heatmap", fill=(148, 163, 184))
    card.paste(heatmap_img.resize((120, 120), Image.BICUBIC), (175, 75))

    # Stats block
    draw.text((320, 65), f"Prediction : Digit {pred}", fill=(52, 211, 153))
    draw.text((320, 95), f"Confidence : {conf:.2%}", fill=(241, 245, 249))
    draw.text((320, 125), f"Entropy    : {entropy:.3f} bits", fill=(241, 245, 249))

    # Top 3 candidates
    draw.text((320, 165), "Top Model Candidates:", fill=(148, 163, 184))
    y = 192
    for rank, (digit, prob) in enumerate(top3, 1):
        draw.text((320, y), f"#{rank}  Digit {digit}  —  {prob:.1%}", fill=(203, 213, 225))
        y += 24

    draw.text((16, 312), "Generated with PyTorch CNN | digit_cnn.pt", fill=(100, 116, 139))

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------------------------------------
# App Layout
# -------------------------------------------------------------

model = load_model()

# Header & Badges
st.markdown('<div class="main-title">DigitVision 🧠</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="badge-container">
        <span class="tech-badge">PyTorch 2.x</span>
        <span class="tech-badge">MNIST 99.1% Acc</span>
        <span class="tech-badge">Grad-CAM XAI</span>
        <span class="tech-badge">FGSM Robustness</span>
        <span class="tech-badge">Feature Map Inspector</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #94a3b8; font-size: 1.05rem; margin-top: -8px; margin-bottom: 24px;">'
    "Draw, upload, or benchmark digits against a deep PyTorch CNN. "
    "Explore attribution heatmaps, inspect internal convolutional filters, and test model robustness in real time."
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model Controls")
    xai_method = st.selectbox(
        "Attribution Method",
        options=["gradcam", "saliency", "input_grad"],
        format_func=lambda x: {
            "gradcam": "Grad-CAM (Conv2 Layer)",
            "saliency": "Vanilla Saliency (Gradients)",
            "input_grad": "Input × Gradient",
        }[x],
        help="Grad-CAM highlights discriminative convolutional patterns; Saliency highlights pixel-level sensitivity.",
    )

    cmap_choice = st.selectbox(
        "Heatmap Colormap",
        options=["inferno", "jet", "viridis", "plasma"],
        index=0,
        help="Color spectrum used to render attribution intensities.",
    )

    overlay_alpha = st.slider(
        "Heatmap Overlay Blend",
        min_value=0.1,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Blend factor between input digit and heatmap.",
    )

    st.markdown("---")
    st.header("🔬 Model Architecture")
    st.markdown(
        """
        - **Conv1**: `32 filters, 3x3, ReLU`
        - **Pool1**: `Max-Pool 2x2`
        - **Conv2**: `64 filters, 3x3, ReLU`
        - **Pool2**: `Max-Pool 2x2`
        - **Dropout**: `p = 0.25`
        - **FC1**: `3136 → 128 (ReLU)`
        - **FC2**: `128 → 10 (Softmax)`
        - **Parameters**: `~421,642`
        """
    )
    st.markdown("---")
    st.write("Crafted by **Mohd Faraz**")
    st.write("[GitHub Profile](https://github.com/Faraazz05)")


# Main Content - Input Selector
input_mode = st.radio(
    "Choose Input Source:",
    options=["✏️ Draw on Canvas", "📁 Upload Image", "🎲 MNIST Benchmark"],
    horizontal=True,
)

active_img28 = None
ground_truth_label = None

if input_mode == "✏️ Draw on Canvas":
    col_canvas, col_canvas_meta = st.columns([1.2, 1])
    with col_canvas:
        brush_stroke = st.slider("Brush Size", min_value=10, max_value=32, value=18, step=2)
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=brush_stroke,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        st.caption("Tip: Draw large, centered digits (0–9) on the canvas.")

    with col_canvas_meta:
        if canvas.image_data is not None:
            active_img28 = preprocess_canvas_or_image(canvas.image_data)

elif input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit photo or drawing (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        try:
            raw_img = Image.open(uploaded_file)
            c_u1, c_u2 = st.columns([1, 1])
            with c_u1:
                st.image(raw_img, caption="Original Uploaded Image", width=180)
            with c_u2:
                # Auto-detect if image has light background and invert if necessary
                gray_sample = np.array(raw_img.convert("L"))
                is_bright = gray_sample.mean() > 120
                invert_check = st.checkbox(
                    "Invert colors (White digit on black background)",
                    value=is_bright,
                    help="MNIST requires white digits on black background.",
                )
                if invert_check:
                    gray_sample = 255 - gray_sample
                active_img28 = preprocess_canvas_or_image(gray_sample)
        except Exception as e:
            st.error(f"Error loading image: {e}")

elif input_mode == "🎲 MNIST Benchmark":
    mnist_samples = get_mnist_benchmark_samples()
    if mnist_samples:
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            digit_target = st.selectbox("Select Target Digit:", options=list(range(10)), index=7)
        with b2:
            variant_options = list(range(len(mnist_samples[digit_target])))
            variant = st.selectbox("Sample Variant:", options=variant_options, format_func=lambda x: f"Variant #{x+1}")
        with b3:
            st.write("")
            st.write("")
            random_pick = st.button("🎲 Random Sample")

        if random_pick:
            rand_digit = np.random.randint(0, 10)
            rand_variant = np.random.randint(0, len(mnist_samples[rand_digit]))
            sample_idx = mnist_samples[rand_digit][rand_variant]
            ground_truth_label = rand_digit
        else:
            sample_idx = mnist_samples[digit_target][variant]
            ground_truth_label = digit_target

        ds = datasets.MNIST(str(DATA_DIR), train=False, download=False)
        active_img28, true_lbl = ds[sample_idx]
        ground_truth_label = true_lbl
        st.success(f"Loaded official MNIST Test Sample #{sample_idx} — Ground Truth: **Digit {ground_truth_label}**")
    else:
        st.info("MNIST raw test data not detected. You can draw digits using the canvas!")


# -------------------------------------------------------------
# Results & Deep Visualizations
# -------------------------------------------------------------

st.markdown("---")

if active_img28 is None:
    st.info("💡 Draw a digit, upload an image, or select an MNIST sample above to view live analysis.")
else:
    # 1. Inference
    t0 = time.perf_counter()
    arr = np.array(active_img28, dtype="float32") / 255.0
    x_tensor = torch.tensor((arr - MEAN) / STD, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = logits.argmax(1).item()
        conf = probs[pred]

    latency_ms = (time.perf_counter() - t0) * 1000

    # Shannon Entropy H(p) = -sum p * log2(p)
    entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)))

    # Compute XAI attribution map
    xai_map = compute_xai_map(model, x_tensor, pred, method=xai_method)

    # Blend Heatmap onto Base Image
    base_rgb = np.stack([arr] * 3, axis=-1)
    color_heat = apply_colormap(xai_map, cmap=cmap_choice) / 255.0
    blended = np.clip((1.0 - overlay_alpha) * base_rgb + overlay_alpha * color_heat, 0.0, 1.0)
    blended_img = Image.fromarray((blended * 255).astype(np.uint8))
    pure_heatmap_img = Image.fromarray((color_heat * 255).astype(np.uint8))

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Predicted Digit", f"{pred}", delta="Match" if ground_truth_label == pred else None)
    with m_col2:
        st.metric("Model Confidence", f"{conf:.1%}")
    with m_col3:
        if entropy < 0.6:
            entropy_label = "Low (Decisive) 🟢"
        elif entropy < 1.4:
            entropy_label = "Moderate 🟡"
        else:
            entropy_label = "High (Ambiguous) 🔴"
        st.metric("Shannon Entropy", f"{entropy:.2f} bits", delta=entropy_label, delta_color="off")
    with m_col4:
        st.metric("Inference Latency", f"{latency_ms:.1f} ms")

    # Visual Inspection Columns
    st.subheader("🔍 Visual Attribution & Explainability")
    v1, v2, v3 = st.columns(3)

    with v1:
        st.image(
            active_img28.resize((224, 224), Image.NEAREST),
            caption="What the Model Sees (28×28 Grayscale)",
            use_container_width=True,
        )

    with v2:
        st.image(
            pure_heatmap_img.resize((224, 224), Image.BICUBIC),
            caption=f"Attribution Map ({xai_method.upper()} / {cmap_choice.title()})",
            use_container_width=True,
        )

    with v3:
        st.image(
            blended_img.resize((224, 224), Image.BICUBIC),
            caption=f"Heatmap Overlay (α = {overlay_alpha:.2f})",
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # Feature 1: Adversarial Robustness Tester (FGSM Attack)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🛡️ Adversarial Robustness Simulator (FGSM Attack)")
    st.write(
        "Test how resilient the CNN is to targeted adversarial noise using the **Fast Gradient Sign Method (FGSM)**. "
        "Move the perturbation slider to inject gradient noise into your digit in real time."
    )

    epsilon = st.slider("Perturbation Strength (ε)", min_value=0.00, max_value=0.30, value=0.00, step=0.02)

    if epsilon > 0.0:
        perturbed_x, noise_mask, adv_pred, adv_conf = run_fgsm_attack(model, x_tensor, pred, epsilon)
        # Convert perturbed tensor back to PIL for display
        perturbed_arr = (perturbed_x[0, 0].cpu().numpy() * STD) + MEAN
        perturbed_arr = np.clip(perturbed_arr, 0.0, 1.0)
        perturbed_img = Image.fromarray((perturbed_arr * 255).astype(np.uint8))

        # Noise display
        noise_norm = (noise_mask - noise_mask.min()) / (noise_mask.max() - noise_mask.min() + 1e-8)
        noise_img = Image.fromarray((noise_norm * 255).astype(np.uint8))

        a1, a2, a3 = st.columns(3)
        with a1:
            st.image(noise_img.resize((224, 224), Image.NEAREST), caption="Adversarial Noise Pattern sign(∇x L)", use_container_width=True)
        with a2:
            st.image(perturbed_img.resize((224, 224), Image.NEAREST), caption=f"Perturbed Image (ε = {epsilon})", use_container_width=True)
        with a3:
            if adv_pred != pred:
                st.error(f"⚠️ **Model Fooled!** Prediction flipped to **Digit {adv_pred}** ({adv_conf:.1%} confidence).")
                st.write("The adversarial attack shifted the classification boundary while preserving human readability!")
            else:
                st.success(f"🛡️ **Robust Prediction!** Model still correctly identifies **Digit {pred}** ({adv_conf:.1%} confidence).")
                st.write("The network successfully resists this level of perturbation.")

    # ---------------------------------------------------------
    # Feature 2: Inside the CNN Layers (Feature Maps)
    # ---------------------------------------------------------
    with st.expander("🔬 Inside the Neural Network (Convolutional Feature Maps)", expanded=False):
        st.markdown(
            "See the hidden activations inside each layer. "
            "**Conv1** learns simple visual primitives like edges and stroke angles, "
            "while **Conv2** combines them into complex curves, loops, and junctions."
        )
        c1_maps, c2_maps = extract_layer_feature_maps(model, x_tensor)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.write("**Conv Layer 1 Filters (First 8 of 32)**")
            grid1 = build_feature_grid_image(c1_maps, num_filters=8, cols=4)
            st.image(grid1, caption="Low-level edge & stroke orientation detectors", use_container_width=True)

        with f_col2:
            st.write("**Conv Layer 2 Filters (First 8 of 64)**")
            grid2 = build_feature_grid_image(c2_maps, num_filters=8, cols=4)
            st.image(grid2, caption="High-level curve, junction & loop detectors", use_container_width=True)

    # ---------------------------------------------------------
    # Feature 3: Probability Breakdown & Top-3 Candidates
    # ---------------------------------------------------------
    st.markdown("---")
    c_prob1, c_prob2 = st.columns([1.2, 1])

    with c_prob1:
        st.subheader("📊 Class Probability Distribution")
        chart_data = pd.DataFrame({"Probability": probs}, index=[str(i) for i in range(10)])
        st.bar_chart(chart_data)

    with c_prob2:
        st.subheader("🏆 Top-3 Ranked Candidates")
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_tuples = [(idx, float(probs[idx])) for idx in top3_indices]

        for rank, (d_idx, d_prob) in enumerate(top3_tuples, 1):
            st.write(f"**#{rank} — Digit {d_idx}** ({d_prob:.1%})")
            st.progress(float(d_prob))

    # ---------------------------------------------------------
    # Feature 4: Downloadable Diagnostic Report
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Diagnostic Report")
    card_bytes = generate_diagnostic_card(active_img28, blended_img, pred, conf, entropy, top3_tuples)

    st.download_button(
        label="Download Analysis Summary Card (PNG)",
        data=card_bytes,
        file_name=f"digitvision_pred_{pred}.png",
        mime="image/png",
        help="Download an exportable visual card for presentations, resumes, or reports.",
    )
