import os
import shutil
from tqdm import tqdm

# 设置原始路径（根据你实际路径修改）
NORMAL_SRC = "C:/Users/mina/Downloads/1/chest_xray/train/NORMAL"
PNEUMONIA_SRC = "C:/Users/mina/Downloads/1/chest_xray/train/PNEUMONIA"
OUTPUT_DIR = "C:/Users/mina/Downloads/1/chest_xray_balanced/train"

# 读取图像文件列表
normal_files = os.listdir(NORMAL_SRC)
pneumonia_files = os.listdir(PNEUMONIA_SRC)

# 创建目标文件夹
os.makedirs(os.path.join(OUTPUT_DIR, "NORMAL"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "PNEUMONIA"), exist_ok=True)

# 拷贝 Normal 图像：原图 + 复制两次
for fname in tqdm(normal_files, desc="复制 Normal 图像（原图 + 副本 x2）"):
    src = os.path.join(NORMAL_SRC, fname)

    # 原图
    shutil.copy2(src, os.path.join(OUTPUT_DIR, "NORMAL", fname))

    # 副本1
    shutil.copy2(src, os.path.join(OUTPUT_DIR, "NORMAL", f"copy1_{fname}"))

    # 副本2
    shutil.copy2(src, os.path.join(OUTPUT_DIR, "NORMAL", f"copy2_{fname}"))

# 拷贝所有 Pneumonia 图像
for fname in tqdm(pneumonia_files, desc="复制原始 Pneumonia"):
    src = os.path.join(PNEUMONIA_SRC, fname)
    dst = os.path.join(OUTPUT_DIR, "PNEUMONIA", fname)
    shutil.copy2(src, dst)

print(f"✅ 平衡训练集构建完成：Normal={len(normal_files)*3}, Pneumonia={len(pneumonia_files)}")
