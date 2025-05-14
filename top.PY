import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_topk(
    train_feature_path="train_features.npy",
    train_label_path="train_labels.npy",
    query_feature_path="query_feature.npy",
    topk_distances=None,
    topk_labels=None,
    sample_size=200
):
    """
    可视化 Top-K 检索结果，包括：
    1. 距离柱状图
    2. t-SNE 降维图（query + 训练样本）
    """
    if topk_distances is None:
        topk_distances = [0.9669, 0.9778, 0.9903, 0.9909, 1.0045]
    if topk_labels is None:
        topk_labels = [0, 1, 1, 1, 1]  # 0: Normal, 1: Pneumonia

    # 加载特征与标签
    train_feats = np.load(train_feature_path)
    train_labels = np.load(train_label_path)
    query_feat = np.load(query_feature_path)

    # --- 图 1: Top-K 距离柱状图 ---
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 6), topk_distances, tick_label=[f"Top-{i}" for i in range(1, 6)], color='skyblue')
    plt.title("Query and Top-5 Similar Distances")
    plt.ylabel("Euclidean Distance to Top-5 Similar Images")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("retrieval_vis/topk_distance_bar.png")
    plt.show()

    # --- 图 2: t-SNE 可视化 ---
    sample_indices = np.random.choice(len(train_feats), size=sample_size, replace=False)
    sample_feats = train_feats[sample_indices]
    sample_labels = train_labels[sample_indices]

    combined_feats = np.vstack([sample_feats, query_feat])
    combined_labels = np.concatenate([sample_labels, [-1]])  # -1 表示 query

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_feats)

    plt.figure(figsize=(8, 6))
    for label in [0, 1]:
        idx = combined_labels == label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                    label='Normal' if label == 0 else 'Pneumonia', alpha=0.6)

    # Query 图像
    plt.scatter(tsne_results[-1, 0], tsne_results[-1, 1], c='red', 
                label='Query Image', edgecolor='black', s=120, marker='*')

    plt.legend()
    plt.title("t-SNE Visualization: Query vs Training Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("retrieval_vis/tsne_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_topk()
