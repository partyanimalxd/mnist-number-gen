import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- Model and App Configuration ---
# Parameters for the high-efficiency model
LATENT_DIM = 100
N_CLASSES = 10
EMBEDDING_DIM = 10
IMG_SHAPE = (1, 28, 28)

# --- Define the Generator Architecture (Simplified Version) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(N_CLASSES, EMBEDDING_DIM)
        
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBEDDING_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(IMG_SHAPE))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((z, label_emb), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *IMG_SHAPE)
        return img

# --- Streamlit App Core Functions ---
@st.cache_resource
def load_model(weights_path='generator_weights.pth'):
    device = torch.device('cpu')
    model = Generator().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def generate_images(generator_model, digit, num_images=5):
    with torch.no_grad():
        z = torch.randn(num_images, LATENT_DIM)
        labels = torch.LongTensor([digit] * num_images)
        generated_imgs_tensor = generator_model(z, labels)
        generated_imgs_tensor = (generated_imgs_tensor * 0.5) + 0.5
    return generated_imgs_tensor

# --- Main Application UI ---
st.set_page_config(page_title="MNIST Digit Generator", layout="wide")

st.title("Handwritten Digit Generator")
st.write(
    "This app uses a Conditional Generative Adversarial Network to generate "
    "images of handwritten digits. Select a digit and click the button to see the results."
)
st.write("---")

WEIGHTS_FILE = "generator_weights.pth"
if not os.path.exists(WEIGHTS_FILE):
    st.error(f"Model weights file not found! Please make sure `{WEIGHTS_FILE}` is in the same directory.")
    st.stop()

generator = load_model()

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Controls")
    selected_digit = st.selectbox("Select a digit (0-9):", options=list(range(10)), index=7)
    generate_button = st.button("Generate Images", type="primary", use_container_width=True)

with col2:
    st.subheader("Generated Images")
    if 'last_generated_images' not in st.session_state:
        st.session_state.last_generated_images = None

    if generate_button:
        images_tensor = generate_images(generator, selected_digit, num_images=5)
        st.session_state.last_generated_images = [img.squeeze().cpu().numpy() for img in images_tensor]
        st.session_state.last_generated_digit = selected_digit

    if st.session_state.last_generated_images:
        display_cols = st.columns(5)
        for i, image_np in enumerate(st.session_state.last_generated_images):
            with display_cols[i]:
                st.image(
                    image_np,
                    caption=f"Generated {st.session_state.last_generated_digit}",
                    width=128,
                    use_column_width='auto'
                )
    else:
        st.info("Select a digit and click 'Generate Images' to start.")