import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Page config
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image to detect the plant condition.")

# Class names
class_names = [
    "pepper_healthy",
    "potato_early_blight",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight"
]

# Load model
@st.cache_resource
def load_model():
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("best_model_ft.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Upload
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        pred_class = class_names[pred.item()]
        confidence = conf.item() * 100

        st.markdown("### Result")
        st.success(f"Prediction: **{pred_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error("Invalid image. Please upload a valid leaf image.")
        st.write(str(e))
else:
    st.markdown("Upload a clear leaf image to begin.")

