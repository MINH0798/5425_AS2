import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import os
import matplotlib.pyplot as plt
from retrieval import predict_and_search

def evaluate_query_batch(
    test_features_path="retrieval_vis/test_features.npy",
    test_labels_path="retrieval_vis/test_labels.npy",
    test_paths_path="retrieval_vis/test_paths.npy",
    knn_model=None,
    faiss_index=None,
    train_labels=None,
    train_data=None,
    transform=None,
    model=None,
    device=None,
    topk=5,
    save_dir="evaluation_report"
):
    """
    批量评估测试样本的分类准确率、Top-1 精度、Top-K 检索一致性。
    """
    if not os.path.exists(test_features_path):
        raise FileNotFoundError(f"❌ test_features_path not found: {test_features_path}")
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"❌ test_labels_path not found: {test_labels_path}")
    if not os.path.exists(test_paths_path):
        raise FileNotFoundError(f"❌ test_paths_path not found: {test_paths_path}")

    #
    os.makedirs(save_dir, exist_ok=True)

    # 加载测试集
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)
    test_paths = np.load(test_paths_path, allow_pickle=True)

    # 分类准确率
    pred_labels = knn_model.predict(test_features)
    acc = accuracy_score(test_labels, pred_labels)
    print(f"\n✅KNN Classification Accuracy {acc:.4f}")

    top1_correct = 0
    precision_at_5 = []
    report_lines = []

    for i, (feat, true_label, path) in enumerate(zip(test_features, test_labels, test_paths)):
        feat = feat.reshape(1, -1).astype('float32')
        distances, indices = faiss_index.search(feat, topk)
        topk_preds = train_labels[indices[0]]

        top1 = topk_preds[0]
        top1_correct += int(top1 == true_label)

        precision = np.sum(topk_preds == true_label) / topk
        precision_at_5.append(precision)

        report_lines.append(
            f"Test-{i+1}: True={true_label}, Top-1={top1}, P@{topk}={precision:.2f}, Path={path}"
        )

    top1_acc = top1_correct / len(test_labels)
    mean_p_at_5 = np.mean(precision_at_5)

    print(f"✅ Top-1 Accuracy: {top1_acc:.4f}")
    print(f"✅ Mean Precision@{topk}: {mean_p_at_5:.4f}")

    with open(os.path.join(save_dir, "query_batch_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total Samples: {len(test_labels)}\n")
        f.write(f"KNN Accuracy: {acc:.4f}\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
        f.write(f"Mean Precision@{topk}: {mean_p_at_5:.4f}\n\n")
        f.write("\n".join(report_lines))

    # 条形图展示
    plt.figure(figsize=(6, 4))
    plt.bar(["KNN Accuracy", "Top-1 Acc", f"Precision@{topk}"], [acc, top1_acc, mean_p_at_5], color='lightgreen')
    plt.ylim(0, 1.0)
    plt.title("Classification and Retrieval Evaluation")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluation_scores.png"))
    plt.close()
    print("✅ evaluate_query_batch returned:", acc, top1_acc, mean_p_at_5)
    return acc, top1_acc, mean_p_at_5