# DigitVision 🧠 — Deep Learning Digit Classifier & Explainable AI Suite

An interactive computer vision platform built with **PyTorch** and **Streamlit**. DigitVision lets you draw, upload, or benchmark handwritten digits against a convolutional neural network, with real-time explainability (Grad-CAM, Saliency), convolutional feature map inspection, and adversarial robustness testing.

---

## 🎥 Demo Walkthrough

<div align="center">
  <video src="demo.mp4" controls="controls" width="100%"></video>
</div>

> 📹 *If the video player doesn't load directly in your preview, click [here to view demo.mp4](demo.mp4).*

---

## ✨ Key Features

1. **🎨 Multi-Modal Input Suite**
   - **Interactive Freehand Canvas**: Draw digits with adjustable brush sizes.
   - **Image Upload**: Upload photos or scans of handwritten numbers with automatic background detection and color inversion.
   - **MNIST Benchmark Gallery**: Test benchmark digits directly from the official MNIST test dataset with ground truth labels.

2. **🔍 Explainable AI (XAI) Engine**
   - **Grad-CAM (Gradient-Weighted Class Activation Mapping)**: Visualizes discriminative high-level activation regions in Conv2.
   - **Vanilla Saliency Maps**: Pixel-level sensitivity gradients demonstrating input sensitivity.
   - **Input × Gradient Attribution**: High-precision attribution maps.
   - **Colormap & Blend Controls**: Switch between *Inferno*, *Jet*, *Viridis*, and *Plasma* palettes with live opacity adjustments.

3. **🛡️ Adversarial Robustness Simulator (FGSM)**
   - Test the CNN's resilience to targeted adversarial perturbations using the **Fast Gradient Sign Method (FGSM)**:
     $$\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \cdot \operatorname{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, y))$$
   - Live perturbation mask visualization and vulnerability thresholding.

4. **🔬 Convolutional Feature Map Inspector**
   - Peek inside the neural network:
     - **Conv1 (32 filters)**: Low-level edge and stroke orientation detectors.
     - **Conv2 (64 filters)**: High-level curve, junction, and loop detectors.

5. **📊 Uncertainty Diagnostics & Exportable Report**
   - **Shannon Entropy Metric** ($H(p) = -\sum p_i \log_2 p_i$): Classifies model confidence as *Decisive*, *Moderate*, or *Ambiguous*.
   - **Top-3 Ranked Candidates**: Ranked probability progress bars.
   - **📥 Export Diagnostic Card**: One-click generation of an exportable PNG summary card containing predictions, attribution heatmaps, and metrics.

---

## 🛠️ Tech Stack & Model Specs

- **Frameworks**: PyTorch, Torchvision, Streamlit, NumPy, Pillow, Pandas.
- **Model Architecture**:
  - `Conv2D(1, 32, kernel=3, padding=1) + ReLU`
  - `MaxPool2D(2, 2)`
  - `Conv2D(32, 64, kernel=3, padding=1) + ReLU`
  - `MaxPool2D(2, 2)`
  - `Dropout(0.25)`
  - `Linear(3136 → 128) + ReLU`
  - `Linear(128 → 10)`
- **Parameters**: ~421,642
- **Test Accuracy**: ~99.1% on MNIST test set.

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Faraazz05/DigitVision.git
cd DigitVision
pip install -r requirements.txt
```

### 2. Train the Model (Optional — Pre-trained Weights Included)
```bash
python train.py
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

