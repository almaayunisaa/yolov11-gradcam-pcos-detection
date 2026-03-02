# Import Library
import cv2
import numpy as np
import time
from PIL import Image
import io
import streamlit as st
import os
import torch
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

model = YOLO(model_path)
model.model.float()
model.model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

PCOS_CLASS_ID   = 1
NORMAL_CLASS_ID = 0

THRESHOLD       = 0.2
TOPK             = 50

class_mapping = {0: "Normal", 1: "PCOS"}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (640, 640))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

target_layer = model.model.model[-3]
gradcam = GradCAM(model.model, target_layer)

def preprocess_image(img):
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(True)
    return img_tensor

def run_gradcam(img):
    if img is None:
        return None

    img = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = preprocess_image(img_rgb)

    preds = model.model(img_tensor)
    output = preds[0]
    raw_logits = output[0, 4:, :]

    conf_normal = float(raw_logits[0, :].max().detach())
    conf_pcos   = float(raw_logits[1, :].max().detach())

    PCOS_THRESHOLD = 0.15
    if conf_pcos >= PCOS_THRESHOLD:
        pred_id    = 1
        confidence = conf_pcos
    else:
        pred_id    = 0
        confidence = conf_normal

    pred_label = class_mapping.get(pred_id, "Unknown")

    target_logits = raw_logits[pred_id, :]
    topk_values, _ = torch.topk(target_logits, k=TOPK)
    score = topk_values.sum()

    model.model.zero_grad()
    score.backward()

    # Generate CAM
    cam = gradcam.generate()
    cam[cam < THRESHOLD] = 0 

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # JET seragam
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    result_img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    text = f"Pred: {pred_label} ({confidence:.2f})"
    cv2.putText(result_img, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(result_img, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    return result_img

def preprocess_ori(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    clahe_img = clahe.apply(image_bw)
    image_resized = cv2.resize(clahe_img, (640, 640))
    final_img = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    return final_img

if 'new_img' not in st.session_state:
    st.session_state['new_img'] = None

if 'original_img' not in st.session_state:
    st.session_state['original_img'] = None

if 'time' not in st.session_state:
    st.session_state['time'] = None

st.header("Explainable PCOS YOLOv11 Detection with Grad-CAM")

st.subheader("Upload Citra")
file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_clahe = preprocess_ori(img)

    start_time = time.time()
    result = run_gradcam(img_clahe)
    end = time.time()

    st.session_state['new_img'] = result
    st.session_state['original_img'] = img_clahe
    st.session_state['time'] = end - start_time

    if st.button("Reset Proccess"):
        st.session_state['new_img'] = None
        st.session_state['original_img'] = None
        st.session_state['time'] = None
        st.stop()

if st.session_state['new_img'] is not None:
    st.success(f"Grad-CAM Berhasil dengan waktu {st.session_state['time']:.2f} detik")

    st.text("Before Grad-CAM (CLAHE)")
    st.image(st.session_state['original_img'], caption="Before")

    st.text("After Grad-CAM")
    st.image(st.session_state['new_img'], caption="After")

    img_dw = Image.fromarray(st.session_state['new_img'])
    buffer = io.BytesIO()
    img_dw.save(buffer, format=file.type.split("/")[1].upper())
    buffer.seek(0)

    st.download_button(
        label="Download Grad-CAM Image",
        data=buffer,
        file_name=file.name.split('.')[0] + "_gradcam" + "." + file.type.split("/")[1],
        mime=file.type,
        icon=":material/download:",
    )