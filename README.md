# Handwritten Digit Generator Web App

A web application that generates handwritten digits (0-9) using a trained Conditional GAN model.

## Features

- Select any digit (0-9) to generate
- Generate 5 unique images of the selected digit
- Interactive web interface built with Streamlit
- Trained on MNIST dataset using PyTorch

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app locally:
```bash
streamlit run app.py
```

## Files

- `app.py` - Main Streamlit application
- `digit_generator.pth` - Trained model weights
- `model.ipynb` - Training notebook
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Model Architecture

The app uses a Conditional Generative Adversarial Network (cGAN) with:
- Generator: Takes 20D noise vector + 10D one-hot label → 28×28 image
- Trained on MNIST dataset
- Framework: PyTorch

## Deployment

This app can be deployed on Streamlit Community Cloud. See deployment instructions below.
