import numpy as np
import faiss
import pandas as pd
from sklearn.preprocessing import normalize
import os

def batch_retrieval_report(
    test_features_path="test_features.npy",
    test_labels_path="test_labels.npy",
    train_features_path="train_features.npy",
    train_labels_path="train_labels.npy",
    topk=5,
    save_path="retrieval_vis/topk_report.csv"
):
    """
    批量评估每张测试图像的 Top-K 检索结果，并输出标签分布与精度
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 加载数据并归一化
    test_feats = normalize(np.load(test_features_path), norm='l2')
    test_labels = np.load(test_labels_path)
    train_feats = normalize(np.load(train_features_path), norm='l2')
    train_labels = np.load(train_labels_path)

    # 构建 FAISS 内积索引（cosine similarity）
    dim = train_feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_feats.astype('float32'))

    # 遍历测试集
    records = []
    for i, (feat, true_label) in enumerate(zip(test_feats, test_labels)):
        feat = feat.reshape(1, -1).astype('float32')
        faiss.normalize_L2(feat)
        D, I = index.search(feat, topk)
        topk_preds = train_labels[I[0]]

        precision = np.sum(topk_preds == true_label) / topk
        top1_correct = (topk_preds[0] == true_label)

        records.append({
            "Index": i,
            "True Label": int(true_label),
            "Top-1 Correct": int(top1_correct),
            "Precision@%d" % topk: precision,
            "Top-%d Labels" % topk: ", ".join(map(str, topk_preds))
        })

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ Top-K 检索分析已保存至: {save_path}")

if __name__ == "__main__":
    batch_retrieval_report()