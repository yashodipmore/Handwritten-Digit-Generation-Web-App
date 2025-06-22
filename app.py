import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Define the Generator model (same as in training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20 + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.fc(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    device = torch.device("cpu")  # Use CPU for deployment
    generator = Generator()
    generator.load_state_dict(torch.load("digit_generator.pth", map_location=device))
    generator.eval()
    return generator

def one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    return torch.nn.functional.one_hot(torch.tensor(labels), num_classes).float()

def generate_digits(generator, digit, num_images=5):
    """Generate specified number of digit images"""
    with torch.no_grad():
        # Create noise vectors
        z = torch.randn(num_images, 20)
        # Create labels (all same digit)
        labels = [digit] * num_images
        labels_oh = one_hot(labels, 10)
        
        # Generate images
        fake_imgs = generator(z, labels_oh)
        
        # Convert to numpy and scale to 0-255
        imgs = fake_imgs.cpu().numpy()
        imgs = (imgs + 1) / 2  # Scale from [-1, 1] to [0, 1]
        imgs = (imgs * 255).astype(np.uint8)
        
        return imgs

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle("Generated Handwritten Digits", fontsize=16, fontweight='bold')
    
    for i, img in enumerate(images):
        axes[i].imshow(img[0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    return fig

# Streamlit App
def main():
    st.set_page_config(
        page_title="Handwritten Digit Generator",
        page_icon="✏️",
        layout="wide"
    )
    
    # Title and description
    st.title("✏️ Handwritten Digit Generator")
    st.markdown("""
    This web app generates handwritten digits (0-9) using a trained Conditional GAN model.
    Select a digit below and click 'Generate' to create 5 unique images of that digit!
    """)
    
    # Load model
    try:
        generator = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please make sure 'digit_generator.pth' is in the same directory as this app.")
        return
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Choose a digit to generate:",
        options=list(range(10)),
        index=0,
        help="Select which digit (0-9) you want to generate"
    )
    
    # Generate button
    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating handwritten digits..."):
            try:
                # Generate images
                generated_images = generate_digits(generator, selected_digit, 5)
                
                # Create and display the grid
                fig = create_image_grid(generated_images)
                st.pyplot(fig)
                plt.close(fig)  # Close to free memory
                
                # Display individual images in columns
                st.subheader(f"Generated Images of Digit: {selected_digit}")
                cols = st.columns(5)
                
                for i, img in enumerate(generated_images):
                    with cols[i]:
                        st.image(img[0], caption=f"Image {i+1}", use_column_width=True, clamp=True)
                
                st.success(f"✅ Successfully generated 5 images of digit {selected_digit}!")
                
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This app uses a Conditional Generative Adversarial Network (cGAN) 
    trained on the MNIST dataset to generate handwritten digits.
    
    **Model Details:**
    - Framework: PyTorch
    - Architecture: Conditional GAN
    - Input: Random noise + digit label
    - Output: 28×28 grayscale images
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit 🚀 | Powered by PyTorch ⚡"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
