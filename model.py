import torch
from ultralytics import YOLO
import easyocr
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import os


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefer a local segmentation model; fall back to other known names.
    if os.path.exists("yolo11n-seg.pt"):
        yolo_source = "yolo11n-seg.pt"
    elif os.path.exists("yolo11n.pt"):
        yolo_source = "yolo11n.pt"
    else:
        yolo_source = "yolo11n-seg.pt"

    yolo_model = YOLO(yolo_source).to("cpu")

    ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model.to(device)

    return yolo_model, ocr_reader, processor, caption_model     





