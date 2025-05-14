import os
import numpy as np
import torch  # ✅ 加上这行导入 torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from evaluate import evaluate_query_batch   # ✅ 导入 evaluate_query_batch 函数

from model import HybridFeatureExtractor
from utils import get_transform
from retrieval import extract_features, build_faiss_index, build_knn_classifier, predict_and_search




if __name__ == "__main__":
    base_path = "./chest_xray_balanced"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    train_data = ImageFolder(root=os.path.join(base_path, "train"), transform=transform)
    test_data = ImageFolder(root=os.path.join(base_path, "test"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True)

    model = HybridFeatureExtractor().to(device)
    model.eval()

    extract_features(train_loader, model, device, "train")
    extract_features(test_loader, model, device, "test")

    train_feats = np.load("train_features.npy").astype('float32')
    train_labels = np.load("train_labels.npy")
    test_feats = np.load("test_features.npy").astype('float32')
    test_labels = np.load("test_labels.npy")

    index = build_faiss_index(train_feats)
    knn = build_knn_classifier(train_feats, train_labels, k=5)
    custom_image_path = "./1.jpeg"  # 👈 把你想预测的图像放到这个位置，并修改名称
   

    
    # ✅ 处理上传图像
    if os.path.exists(custom_image_path):
        print(f"\n🚀 开始使用上传图像进行预测和检索...")
        predict_and_search(custom_image_path, model, knn, index, train_labels, train_data, transform, device, topk=5, save_visual=True)
    else:
        print("❌ 找不到上传图像，请确认路径是否正确！")


    evaluate_query_batch(
    test_features_path="test_features.npy",
    test_labels_path="test_labels.npy",
    test_paths_path="test_paths.npy",
    knn_model=knn,
    faiss_index=index,
    train_labels=train_labels,
    train_data=train_data,
    transform=transform,
    model=model,
    device=device,
    topk=5
)
    