import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册 Hook
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        input_image = input_image.to(next(self.model.parameters()).device)
        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = pooled_gradients[None, :, None, None] * self.activations
        heatmap = torch.sum(weighted_activations, dim=1).squeeze().cpu().numpy()

        # ReLU + normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))

        return heatmap

def apply_heatmap_on_image(image_pil, heatmap):
    """
    将 Grad-CAM 生成的热力图叠加到原图上，返回彩色叠加图像（numpy array）
    """
    image = np.array(image_pil.resize((heatmap.shape[1], heatmap.shape[0])))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.5, image, 0.5, 0)
    return overlay
