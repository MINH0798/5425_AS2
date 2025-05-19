import streamlit as st
st.set_page_config(page_title="Medical Image Retrieval System", layout="wide")

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
from collections import Counter




def display_retrieved_images(topk_results, image_size=(224, 224),  small=False):
    if small:
        st.markdown("🖼️ **Top-5 Retrieved Images:**")
    else:
        st.subheader("🖼️ Top-5 Retrieved Images")
    
    st.caption("All retrieved images are resized to 224×224 for consistent display.")
    cols = st.columns(len(topk_results))
    for i, (path, label, dist) in enumerate(topk_results):
        with cols[i]:
            img = Image.open(path).convert("RGB").resize(image_size)
            st.image(img, caption=f"Top-{i+1} {label} ({dist:.4f})", use_container_width=False)









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

@st.cache_data
def cached_evaluate():
    return evaluate_query_batch(
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

with st.expander("🧹 Clear All Retrieval Results (Click to confirm)"):
    confirm_clear = st.checkbox("✅ Yes, I want to clear all previous retrieval results.")
    if confirm_clear:
        if st.button("🚨 Confirm & Clear Now"):
            for key in list(st.session_state.keys()):
                if key.startswith("result_"):
                    del st.session_state[key]
            st.rerun()




# ✅ 上传图片后处理,多次传图
uploaded_files = st.file_uploader("📤 Upload image(s) (JPEG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with st.form(key=f"form_{idx}"):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image.resize((224, 224)), caption="Uploaded Image", use_container_width=False, width=224)
            submit = st.form_submit_button("🔍 Start Retrieval")
            if submit:
                # === Grad-CAM 热力图展示 ===
                st.subheader("🔥 Grad-CAM Heatmap")
                input_tensor = transform(image).unsqueeze(0).to(device)
                heatmap = gradcam.generate(input_tensor)
                overlay = apply_heatmap_on_image(image, heatmap)
                timestamp = int(time.time())

                gradcam_path = os.path.join("retrieval_vis", f"gradcam_overlay_{idx}_{timestamp}.jpg")

                cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                st.image(Image.open(gradcam_path).resize((224, 224)), caption="Grad-CAM Visualization", use_container_width=False, width=224)

            # 保存上传图像副本
                query_path = os.path.join("retrieval_vis", f"query_image_{idx}_{timestamp}.png")
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
                knn_pred_label = class_names[pred_label]                      # 多数投票结果
                top1_label = topk_results[0][1]                               # Top-1 最近邻标签
                label_counter = Counter([label for (_, label, _) in topk_results]) # 标签统计
                sorted_labels = sorted(label_counter.items(), key=lambda x: -x[1])
                distribution_str = ", ".join([f"{label}: {count}" for label, count in sorted_labels])
                st.markdown(f"📊 **Top-5 Label Distribution:** `{distribution_str}`")

                # ✅ 显示 KNN 多数投票预测
                st.success(f"✅ **Predicted Class (KNN Majority)**: `{knn_pred_label}`")

                # 🔍 显示 Top-1 最近邻标签
                st.info(f"🔍 **Top-1 Nearest Label**: `{top1_label}`")

                st.write(f"⏱️ Retrieval Time: {elapsed:.4f} seconds")
                # st.subheader("📊 Top-5 Retrieved Images Path")
                # for i, (path, label, dist) in enumerate(topk_results):
                #     st.write(f"Top-{i+1}: {path} | Label: {label} | Dist: {dist:.4f}")
                
               
                st.subheader("📊 Top-5 Retrieved Images Path")
                for i, (path, label, dist) in enumerate(topk_results):
                    abs_path = os.path.abspath(path)
                    abs_path = abs_path.replace("\\", "/")  # Windows 路径兼容性
                    file_url = f"file:///{abs_path}"        # 注意多一个 / 以符合 URL 规范
                    filename = os.path.basename(path)
                    st.markdown(
                        f"🔗 **Top-{i+1}:** [{filename}]({file_url}) | **Label:** `{label}` | **Dist:** `{dist:.4f}`",
                        unsafe_allow_html=True
                    )
                    # st.caption(f"🛠️ Debug: {abs_path}")

                display_retrieved_images(topk_results)
                # st.subheader("🖼️ Top-5 Retrieved Images")
                # cols = st.columns(5)
                # for i, (path, label, dist) in enumerate(topk_results):
                #     with cols[i]:
                #         st.image(path, use_container_width=True, caption=f"Top-{i+1} {label} ({dist:.4f})")

                # st.image("retrieval_vis/topk_result.png", caption="Top-5 Retrieved Images", use_container_width=True) 展示合并的图片
                st.session_state[f"result_{idx}"] = {
                    "gradcam_path": gradcam_path,
                    "topk_results": topk_results,
                    "knn_pred_label": knn_pred_label,
                    "top1_label": top1_label,
                    "distribution_str": distribution_str,
                    "elapsed": elapsed
                }

            # 批量评估
    st.subheader("📈 Batch Evaluation Metrics")
    with st.spinner("Evaluating overall test set performance..."):
        acc_knn, acc_top1, mean_p5 = cached_evaluate()

    st.success(f"✅ KNN Classification Accuracy: {acc_knn:.4f}")
    st.success(f"✅ Top-1 Accuracy: {acc_top1:.4f}")
    st.success(f"✅ Mean Precision@5: {mean_p5:.4f}")


st.markdown("## 📂 Retrieval History")

for key in sorted(st.session_state.keys()):
    if key.startswith("result_"):
        result = st.session_state[key]
        idx = key.replace("result_", "")

        with st.expander(f"📁 Retrieval Result for Image #{idx}", expanded=False):
            # Grad-CAM 可视化
            st.image(
                Image.open(result["gradcam_path"]).resize((224, 224)),
                caption="Grad-CAM",
                use_container_width=False,
                width=224
            )

            # 分布 & 预测标签
            st.markdown(f"📊 **Top-5 Label Distribution:** `{result['distribution_str']}`")
            st.success(f"✅ **Predicted Class (KNN Majority):** `{result['knn_pred_label']}`")
            st.info(f"🔍 **Top-1 Nearest Label:** `{result['top1_label']}`")
            st.write(f"⏱️ Retrieval Time: {result['elapsed']:.4f} seconds")

            # 路径 + 超链接
            st.markdown("📊 **Top-5 Retrieved Images Path:**")

            for i, (path, label, dist) in enumerate(result["topk_results"]):
                abs_path = os.path.abspath(path).replace("\\", "/")
                folder_path = os.path.dirname(abs_path)
                folder_url = f"file:///{folder_path}"
                file_url = f"file:///{abs_path}"
                filename = os.path.basename(path)

                st.markdown(
                    f"🔗 **Top-{i+1}:** `{filename}` | "
                    f"📁 [Open Folder]({folder_url}) | "
                    f"🖼️ [Open Image]({file_url}) | "
                    f"**Label:** `{label}` | **Dist:** `{dist:.4f}`",
                    unsafe_allow_html=True
                )

            # 展示 Top-5 图像
            display_retrieved_images(result["topk_results"], small=True)


