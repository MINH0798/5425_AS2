# 📁 utils.py
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_xray_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    img_denoised = cv2.fastNlMeansDenoising(img_resized, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_contrast = clahe.apply(img_denoised)
    img_normalized = img_contrast / 255.0
    img_3channel = np.stack([img_normalized]*3, axis=-1)
    return img_3channel

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def load_and_transform_image(image_path, transform):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    return image, image_tensor