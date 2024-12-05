import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# 设置数据集路径
dataset_dir = "../Insects_image_dataset/Ant"

# 获取所有图片路径
dataset = []
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            dataset.append(os.path.join(root, file))

print(f"共加载 {len(dataset)} 张图片")

# 加载预训练模型
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 安全加载图片
def open_image_safely(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # 转换为 RGB，忽略 EXIF 数据
        return img
    except Exception as e:
        print(f"无法加载图片 {image_path}: {e}")
        return None

# 分类图片
def classify_image(image_path):
    image = open_image_safely(image_path)
    if image is None:
        return None, None
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        confidence, class_id = torch.max(output, 1)
        return confidence.item(), class_id.item()

# 设置置信度阈值
confidence_threshold = 0.5

irrelevant_images = []  # 存储可能无关的图片

print("开始筛选无关图片...")
for image_path in dataset:
    confidence, class_id = classify_image(image_path)
    if confidence is not None and confidence < confidence_threshold:
        irrelevant_images.append(image_path)
        print(f"可能的无关图片: {image_path}，置信度: {confidence:.2f}")

print(f"检测到 {len(irrelevant_images)} 张可能的无关图片")

# # 可选：清理无关图片
# delete = input("是否删除检测到的无关图片？(y/n): ").strip().lower()
# if delete == "y":
#     for image_path in irrelevant_images:
#         try:
#             os.remove(image_path)
#             print(f"已删除图片: {image_path}")
#         except Exception as e:
#             print(f"无法删除图片 {image_path}: {e}")

# print("处理完成。")