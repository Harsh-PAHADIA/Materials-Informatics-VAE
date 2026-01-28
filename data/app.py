import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from train import MicroVAE 
import pandas as pd

# Page Setup
st.set_page_config(page_title="Materials Informatics Portal", layout="wide")
st.title("🔬 Advanced Microstructural Synthesis & Property Prediction")
st.markdown("---")

# Model Loading
@st.cache_resource
def load_trained_model():
    model = MicroVAE(latent_dim=16)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

model = load_trained_model()

# Sidebar: Professional Labels based on Steel Informatics
st.sidebar.header("Microstructure Control")
f1 = st.sidebar.slider("Grain Refinement (Feature 1)", -3.0, 3.0, 0.0)
f2 = st.sidebar.slider("Phase Density (Feature 2)", -3.0, 3.0, 0.0)
f3 = st.sidebar.slider("Carbide Distribution (Feature 3)", -3.0, 3.0, 0.0)
f4 = st.sidebar.slider("Structural Noise (Feature 4)", -3.0, 3.0, 0.0)

# Latent vector setup for VAE
latent_vector = [f1, f2, f3, f4] + [0.0]*12

# 1. Image Generation via Decoder
with torch.no_grad():
    z = torch.tensor([latent_vector], dtype=torch.float32)
    generated_img = model.decoder(model.decoder_input(z).view(-1, 64, 16, 16))
    img_np = generated_img.squeeze().numpy()

# 2. Multi-Property Prediction Logic
# Hall-Petch based Yield Strength logic
predicted_yield = 320 + (f1 * -45) + (f2 * 10) 
# Hardness logic
predicted_hardness = 180 + (f2 * 25) + (f3 * 15)
# Ductility trade-off logic
predicted_ductility = 25 - (f1 * -2) - (f2 * 3) - (f3 * 4)

# UI Layout: Micrograph and Metrics
col_img, col_metrics = st.columns([2, 1])

with col_img:
    st.markdown("#### Synthetic SEM Micrograph")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    st.pyplot(fig)

with col_metrics:
    st.markdown("### Predicted Material Properties")
    st.metric(label="Yield Strength", value=f"{predicted_yield:.2f} MPa")
    st.metric(label="Vickers Hardness", value=f"{predicted_hardness:.1f} HV")
    st.metric(label="Ductility", value=f"{max(5, predicted_ductility):.1f} %")
    
    st.markdown("---")
    st.markdown("### Analysis Summary")
    if f1 < 0:
        st.write("✅ **Fine Grained Structure detected.** Expect high toughness and yield strength due to Hall-Petch effect.")
    else:
        st.write("⚠️ **Coarse Grained Structure.** Risk of lower yield strength and fatigue resistance.")

# --- Graph section with LIVE MARKER ---
st.markdown("---")
st.markdown("#### Physical Property Analysis (Live tracking)")

# 1. Hall-Petch Trend (Fixed Theoretical Rule)
refinement_range = np.linspace(-3, 3, 20)
yield_trend = 320 + (refinement_range * -45) 

# 2. Your Current Material State (Dynamic Point)
# This point now matches your predicted yield formula
current_yield = 320 + (f1 * -45) + (f2 * 10) 

fig_trend, ax_trend = plt.subplots(figsize=(10, 4))

# Plot the fixed trend line
ax_trend.plot(refinement_range, yield_trend, color='#ff4b4b', alpha=0.3, linestyle='--', label='Theory (Hall-Petch)')

# Live marker that moves with your sliders!
ax_trend.scatter(f1, current_yield, color='black', s=150, edgecolors='white', zorder=5, label='Your Current Material')

# Labels and Styling
ax_trend.set_xlabel("Grain Refinement (Latent Variable 1)", fontsize=10)
ax_trend.set_ylabel("Yield Strength (MPa)", fontsize=10)
ax_trend.set_title("Property-Structure Correlation Mapping", fontsize=12)
ax_trend.legend()
ax_trend.grid(True, alpha=0.2)

st.pyplot(fig_trend)