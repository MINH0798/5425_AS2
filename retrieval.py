# 📁 retrieval.py
from sklearn.preprocessing import normalize
import numpy as np
import faiss
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import torch
import time
import matplotlib.pyplot as plt
import os
from utils import load_and_transform_image

# def build_faiss_index(train_features):
#     index = faiss.IndexFlatL2(train_features.shape[1])
#     index.add(train_features)
#     return index
def build_faiss_index(train_feats):
    """
    构建 FAISS 内积索引（基于 Cosine 相似度），并对特征进行 L2 归一化。
    """
    # ✅ 特征归一化（重要！）
    train_feats = normalize(train_feats, norm='l2')

    # ✅ 使用内积（cosine similarity）索引
    dim = train_feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_feats.astype('float32'))

    return index




def build_knn_classifier(train_features, train_labels, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_features, train_labels)
    return classifier

def extract_features(loader, model, device, save_prefix):
    features, labels, paths = [], [], []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (images, lbls) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

            # ✅ 正确收集图像路径（这段是关键！）
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + images.size(0)
            paths.extend(loader.dataset.samples[start_idx:end_idx])

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                print(f"Processing batch {batch_idx+1}/{total_batches} for {save_prefix}")

    if features:
        os.makedirs("retrieval_vis", exist_ok=True)
        np.save(os.path.join("retrieval_vis", f"{save_prefix}_features.npy"), np.concatenate(features, axis=0))
        np.save(os.path.join("retrieval_vis", f"{save_prefix}_labels.npy"), np.concatenate(labels, axis=0))
        np.save(os.path.join("retrieval_vis", f"{save_prefix}_paths.npy"), np.array([p[0] for p in paths]))
    print(f"✅ {save_prefix} features saved to 'retrieval_vis/'")  





def predict_and_search(image_path, model, knn, index, train_labels, train_data,
                       transform, device, topk=5, save_visual=True,
                       return_paths=False, return_details=False):
    print(f"\n📥 上传图像路径: {image_path}")

    # ✅ 读取和处理图像
    image, image_tensor = load_and_transform_image(image_path, transform)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # ✅ 保存上传图像副本
    os.makedirs("retrieval_vis", exist_ok=True)
    upload_copy_path = os.path.join("retrieval_vis", "query_image.png")
    image.save(upload_copy_path)
    print(f"📸 已保存上传图像副本到: {upload_copy_path}")

    # ✅ 提取特征并保存
    with torch.no_grad():
        feat = model(image_tensor).cpu().numpy()

    np.save("retrieval_vis/query_feature.npy", feat)

    # ✅ KNN 预测类别
    pred_label = knn.predict(feat)[0]
    print(f"\n✅ 预测类别: {'Normal' if pred_label == 0 else 'Pneumonia'}")

    # ✅ FAISS 检索
    start = time.time()
    distances, indices = index.search(feat.astype('float32'), topk)
    duration = time.time() - start
    print(f"⏱️ 检索耗时: {duration:.4f} 秒")

    print("\n===== Top-{} 相似图像结果 =====".format(topk))
    result_lines = [f"📥 上传图像路径: {image_path}",
                    f"✅ 预测类别: {'Normal' if pred_label == 0 else 'Pneumonia'}",
                    f"⏱️ 检索耗时: {duration:.4f} 秒", ""]

    # ✅ 初始化返回值列表
    topk_paths = []
    topk_labels = []
    topk_distances = []

    fig, axes = plt.subplots(1, topk, figsize=(topk * 3, 3)) if save_visual else (None, None)

    for i, idx in enumerate(indices[0]):
        label = train_labels[idx]
        path = train_data.samples[idx][0]
        label_str = 'Normal' if label == 0 else 'Pneumonia'
        dist = distances[0][i]

        topk_paths.append(path)
        topk_labels.append(label_str)
        topk_distances.append(dist)

        result_line = f"Top-{i+1}: {path} | Label: {label_str} | Dist: {dist:.4f}"
        result_lines.append(result_line)
        print(result_line)

        if save_visual:
            img = Image.open(path).convert('RGB')
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Top-{i+1}\n{label_str}")

    if save_visual:
        plt.tight_layout()
        plt.savefig("retrieval_vis/topk_result.png")
        plt.close()

    with open("retrieval_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    if return_paths and return_details:
        return list(zip(topk_paths, topk_labels, topk_distances)), pred_label
    elif return_paths:
        return topk_paths
    else:
        return
