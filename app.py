import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="2D FFT Visualizer", layout="wide")
st.title("2D Fourier Transform Visualizer for DIP Assignment")
st.markdown("""
Upload a grayscale or color image.  
Use the buttons below to visualize the **Magnitude Spectrum**, **Phase Spectrum**, generate a **Sinusoidal Grating**, or demonstrate **Jagging (staircase effect)**.
""")

uploaded_file = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

def normalize_image(arr):
    """Normalize numpy array to 0-255 for proper display"""
    arr_norm = arr - arr.min()
    arr_norm = arr_norm / arr_norm.max() * 255
    return arr_norm.astype(np.uint8)

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((256, 256))
    arr = np.array(img)
    st.subheader("Original Image")
    st.image(img, width=200)

    # Compute FFT
    Fshift = np.fft.fftshift(np.fft.fft2(arr))

    # Buttons
    if st.button("Magnitude Spectrum"):
        magnitude = np.log(1 + np.abs(Fshift))
        magnitude = normalize_image(magnitude)
        col1, col2 = st.columns(2)
        with col1:
            st.image(magnitude, width=200, caption="Magnitude Spectrum")
        with col2:
            st.write("")  # empty for spacing

    if st.button("Phase Spectrum"):
        phase = np.angle(Fshift)
        phase = normalize_image(phase)
        col1, col2 = st.columns(2)
        with col1:
            st.image(phase, width=200, caption="Phase Spectrum")
        with col2:
            st.write("")

    if st.button("Sinusoidal Grating"):
        size = 256
        freq = 10
        x = np.arange(size)
        X, Y = np.meshgrid(x, x)
        grating = 128 + 127 * np.sin(2 * np.pi * freq * X / size)
        grating = normalize_image(grating)
        col1, col2 = st.columns(2)
        with col1:
            st.image(grating, width=200, caption="Sinusoidal Grating")
        with col2:
            st.write("")

    if st.button("Jagging Demo"):
        down_size = 32
        small_img = img.resize((down_size, down_size), Image.BILINEAR)
        jagged_img = small_img.resize((256, 256), Image.NEAREST)
        jagged_arr = np.array(jagged_img)

        F_jag = np.fft.fftshift(np.fft.fft2(jagged_arr))
        magnitude_jag = np.log(1 + np.abs(F_jag))
        magnitude_jag = normalize_image(magnitude_jag)

        col1, col2 = st.columns(2)
        with col1:
            st.image(jagged_img, width=200, caption="Jagged Image")
        with col2:
            st.image(magnitude_jag, width=200, caption="Magnitude Spectrum (Jagging)")
