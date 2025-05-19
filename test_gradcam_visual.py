import torch
import cv2
import numpy as np
from PIL import Image
from model import HybridFeatureExtractor
from gradcam import GradCAM
from utils import get_transform
import os

def apply_heatmap_on_image(image_pil, heatmap):
    image = np.array(image_pil.resize((heatmap.shape[1], heatmap.shape[0])))  # Resize to match heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.5, image, 0.5, 0)
    return overlay

if __name__ == "__main__":
    os.makedirs("retrieval_vis", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 加载模型
    model = HybridFeatureExtractor().to(device)
    model.eval()

    # ✅ 指定目标层（EfficientNet 最后一层 block）
    target_layer = model.effnet.blocks[-1]

    # ✅ 构造 GradCAM 对象
    cam = GradCAM(model, target_layer)

    # ✅ 加载图像并转换
    image_path = "1.jpeg"  # ⚠️ 改为你自己的图像路径
    image_pil = Image.open(image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # ✅ 生成热力图
    heatmap = cam.generate(input_tensor)

    # ✅ 可视化叠加
    overlay = apply_heatmap_on_image(image_pil, heatmap)

    # ✅ 保存叠加结果
    output_path = "retrieval_vis/gradcam_overlay.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Grad-CAM 叠加图已保存：{output_path}")
