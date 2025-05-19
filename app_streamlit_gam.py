import streamlit as st
from PIL import Image
import os
import time
import numpy as np
import torch
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from retrieval import predict_and_search, build_faiss_index, build_knn_classifier, extract_features
from model import HybridFeatureExtractor
from utils import get_transform
from evaluate import evaluate_query_batch
from gradcam import GradCAM, apply_heatmap_on_image

st.set_page_config(page_title="Medical Image Retrieval System", layout="wide")
st.title("🔍 Medical Image Retrieval System")

# ✅ 提前加载设备、模型、transform、GradCAM
base_path = "./chest_xray_balanced"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform()

model = HybridFeatureExtractor().to(device)
model.eval()
target_layer = model.effnet.blocks[-1]
gradcam = GradCAM(model, target_layer)

# ✅ 提前加载数据集（只做一次）
train_data = ImageFolder(root=os.path.join(base_path, "train"), transform=transform)
test_data = ImageFolder(root=os.path.join(base_path, "test"), transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True)

# ✅ 提取/加载特征
os.makedirs("retrieval_vis", exist_ok=True)
if not os.path.exists("retrieval_vis/train_features.npy"):
    extract_features(train_loader, model, device, "train")
    extract_features(test_loader, model, device, "test")

train_feats = np.load("retrieval_vis/train_features.npy").astype("float32")
train_labels = np.load("retrieval_vis/train_labels.npy")
test_feats = np.load("retrieval_vis/test_features.npy").astype("float32")
test_labels = np.load("retrieval_vis/test_labels.npy")

index = build_faiss_index(train_feats)
knn = build_knn_classifier(train_feats, train_labels, k=5)

# ✅ 上传图片后处理
uploaded_file = st.file_uploader("📤 Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    # === Grad-CAM 热力图展示 ===
    st.subheader("🔥 Grad-CAM Heatmap")
    input_tensor = transform(image).unsqueeze(0).to(device)
    heatmap = gradcam.generate(input_tensor)
    overlay = apply_heatmap_on_image(image, heatmap)

    gradcam_path = os.path.join("retrieval_vis", "gradcam_overlay.jpg")
    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    st.image(gradcam_path, caption="Grad-CAM Visualization", use_container_width=False, width=300)

    # 保存上传图像副本
    query_path = os.path.join("retrieval_vis", "query_image.png")
    image.save(query_path)

    st.info("🚀 Running prediction and retrieval based on the uploaded image...")

    # 执行检索
    start_time = time.time()
    topk_results, pred_label = predict_and_search(
        image_path=query_path,
        model=model,
        knn=knn,
        index=index,
        train_labels=train_labels,
        train_data=train_data,
        transform=transform,
        device=device,
        topk=5,
        save_visual=True,
        return_paths=True,
        return_details=True
    )
    elapsed = time.time() - start_time

    # 显示预测与检索结果
    class_names = train_data.classes
    label_name = class_names[pred_label]
    st.success(f"✅ Predicted Class: {label_name}")
    st.write(f"⏱️ Retrieval Time: {elapsed:.4f} seconds")

    st.subheader("📊 Top-5 Retrieved Images Path")
    for i, (path, label, dist) in enumerate(topk_results):
        st.write(f"Top-{i+1}: {path} | Label: {label} | Dist: {dist:.4f}")

    st.image("retrieval_vis/topk_result.png", caption="Top-5 Retrieved Images", use_container_width=True)

    # 批量评估
    st.subheader("📈 Batch Evaluation Metrics")
    with st.spinner("Evaluating overall test set performance..."):
        acc_knn, acc_top1, mean_p5 = evaluate_query_batch(
            test_features_path="retrieval_vis/test_features.npy",
            test_labels_path="retrieval_vis/test_labels.npy",
            test_paths_path="retrieval_vis/test_paths.npy",
            knn_model=knn,
            faiss_index=index,
            train_labels=train_labels,
            train_data=train_data,
            transform=transform,
            model=model,
            device=device,
            topk=5,
        )

    st.success(f"✅ KNN Classification Accuracy: {acc_knn:.4f}")
    st.success(f"✅ Top-1 Accuracy: {acc_top1:.4f}")
    st.success(f"✅ Mean Precision@5: {mean_p5:.4f}")
