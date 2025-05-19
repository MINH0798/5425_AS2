
import streamlit as st
from PIL import Image
import os
import time
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from retrieval import predict_and_search, build_faiss_index, build_knn_classifier, extract_features
from model import HybridFeatureExtractor
from utils import get_transform
from evaluate import evaluate_query_batch

st.set_page_config(page_title="Medical Image Retrieval System", layout="wide")
st.title("🔍 Medical Image Retrieval System")

uploaded_file = st.file_uploader("📤 Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    # 保存上传图像
    os.makedirs("retrieval_vis", exist_ok=True)
    query_path = os.path.join("retrieval_vis", "query_image.png")
    image.save(query_path)

    st.info("🚀 Running prediction and retrieval based on the uploaded image...")

    # 加载模型与数据
    base_path = "./chest_xray_balanced"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()

    train_data = ImageFolder(root=os.path.join(base_path, "train"), transform=transform)
    test_data = ImageFolder(root=os.path.join(base_path, "test"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True)

    model = HybridFeatureExtractor().to(device)
    model.eval()

    # 提取特征
    if not os.path.exists("retrieval_vis/train_features.npy"):
        extract_features(train_loader, model, device, "train")
        extract_features(test_loader, model, device, "test")

    # 加载特征
    train_feats = np.load("retrieval_vis/train_features.npy").astype("float32")
    train_labels = np.load("retrieval_vis/train_labels.npy")
    test_feats = np.load("retrieval_vis/test_features.npy").astype("float32")
    test_labels = np.load("retrieval_vis/test_labels.npy")

    # 构建索引和分类器
    index = build_faiss_index(train_feats)
    knn = build_knn_classifier(train_feats, train_labels, k=5)

    # 检索
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

    # 显示结果
    class_names = train_data.classes  # 自动映射 ['PNEUMONIA', 'NORMAL']
    label_name = class_names[pred_label]
    st.success(f"✅ Predicted Class: {label_name}")
    st.write(f"⏱️ Retrieval Time: {elapsed:.4f} seconds")
    st.subheader("📊 Top-5 Retrieved Images Path")
    for i, (path, label, dist) in enumerate(topk_results):
        st.write(f"Top-{i+1}: {path} | Label: {label} | Dist: {dist:.4f}")

    # 展示图像拼图
    st.image("retrieval_vis/topk_result.png", caption="Top-5 Retrieved Images", use_container_width=True)

    # 批量评估
    st.subheader("📈 Batch Evaluation Metrics")
    with st.spinner("Evaluating overall test set performance..."):
        from evaluate import evaluate_query_batch
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
