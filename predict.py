import os
import platform
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import SwinForImageClassification

# 设备配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple MPS"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)  # 获取CUDA设备名称
else:
    device = torch.device("cpu")
    device_name = "CPU"
print(f"Using device: {device} ({device_name})")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Swin Transformer 模型定义
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.swin = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.swin.classifier = nn.Linear(self.swin.config.hidden_size, num_classes)

    def forward(self, x):
        return self.swin(pixel_values=x).logits

def clean_and_get_classes(dataset_dir):
    """
    清理数据集目录并获取类别标签，要求:
    - 每个目录只能包含子目录或图像文件。
    - 图像文件只能出现在叶子节点目录中。
    - 非图像文件和空目录均会被删除。
    - 非叶子节点目录中的图像文件会被删除。

    :param dataset_dir: 数据集的根目录
    :return: 类别标签列表
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory '{dataset_dir}' does not exist. Please check the path.")
    
    def is_image(item):
        """检查文件是否为图片文件"""
        return item.lower().endswith(('.png', '.jpg', '.jpeg'))

    def clean_directory(directory):
        """
        清理目录，删除非规范内容:
        - 删除非叶子节点的图像文件。
        - 删除非图像文件。
        - 删除空目录。
        返回是否保留该目录。
        """
        has_dirs = False
        has_images = False

        # 遍历目录内容
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # 递归清理子目录
                if not clean_directory(item_path):
                    os.rmdir(item_path)  # 删除空目录
                else:
                    has_dirs = True
            elif os.path.isfile(item_path):
                if is_image(item):
                    has_images = True
                else:
                    os.remove(item_path)  # 删除非图像文件

        # 如果当前目录是非叶子节点，删除图像文件
        if has_dirs and has_images:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and is_image(item):
                    os.remove(item_path)
            has_images = False

        # 保留叶子节点目录或非空的子目录
        return has_dirs or has_images

    # 清理根目录
    if not clean_directory(dataset_dir):
        raise ValueError(f"Dataset directory '{dataset_dir}' is empty or does not meet the required structure.")

    # 获取类别标签
    classes = []
    for root, dirs, files in os.walk(dataset_dir):
        if files and all(is_image(file) for file in files):
            # 获取相对路径作为类别标签
            category = os.path.relpath(root, dataset_dir)
            classes.append(category.replace(os.sep, '/'))

    # 检查是否成功提取类别
    if not classes:
        raise ValueError(f"No valid classes found in dataset directory '{dataset_dir}'. "
                         "Ensure it contains image files organized in folders.")
    
    return sorted(classes)

# 加载并初始化模型和类别标签
def load_model_and_classes(model_path, dataset_dir='./dataset'):
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 检查是否包含类别标签
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        classes = checkpoint["classes"]
        print("Loaded classes from checkpoint.")
    else:
        # 如果没有保存类别标签，从指定的数据集路径加载类别
        if os.path.exists(dataset_dir):
            classes = clean_and_get_classes(dataset_dir)
            print(f"Classes not found in checkpoint. Loaded classes from dataset at '{dataset_dir}'.")
        else:
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist. Cannot infer classes.")
    
    # 获取类别数
    num_classes = len(classes)
    # 初始化模型
    model = SwinTransformerModel(num_classes=num_classes).to(device)
    
    # 加载模型权重
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        # 适配直接保存为 state_dict 的情况
        model.load_state_dict(checkpoint)
    
    model.eval()  # 切换到评估模式
    
    print("Model and classes loaded successfully.")
    return model, classes

# 单张图片预测函数
def predict_image(image_path, model, transform, classes, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
    return class_name

# 文件夹预测函数
def predict_folder(folder_path, model, transform, classes, device):

    def is_image(item):
        """检查文件是否为图片文件"""
        return item.lower().endswith(('.png', '.jpg', '.jpeg'))

    predictions = {}
    for root, dirs, files in os.walk(folder_path):
        for item in files:
            # 检查item是否为图像文件
            if is_image(item):
                # 拼接文件路径
                file_path = os.path.join(root, item)
                # 相对路径存储
                rel_path = os.path.relpath(file_path, folder_path) 
                # # 提取文件名，不要格式后缀
                # file_name = os.path.splitext(item)[0]
                try:
                    # 调用文件预测函数
                    class_name = predict_image(file_path, model, transform, classes, device)
                    predictions[rel_path] = class_name
                except Exception as e:
                    print(f"Error predicting {file_name}: {e}")
    return predictions

# 通用预测函数
def predict(input_path, model, transform, classes, device):
    if os.path.isfile(input_path):  # 如果是文件，进行单张图片预测
        class_name = predict_image(input_path, model, transform, classes, device)
        print(f"图片: {input_path}, 预测类别: {class_name}")
    elif os.path.isdir(input_path):  # 如果是文件夹，进行文件夹预测
        predictions = predict_folder(input_path, model, transform, classes, device)
        for file_name, class_name in predictions.items():
            print(f"图片: {file_name}, 预测类别: {class_name}")
    else:
        print(f"无效路径: {input_path}")  # 如果路径无效

# 获取预测数据路径
def get_input_path(base_path="./test_images"):
    while True:
        input_path = input(f"请输入目标文件名(带扩展名)或文件夹路径(默认路径: {base_path}):").strip()

        # 去除两端的引号
        input_path = input_path.strip('"').strip("'")

        # 如果用户未输入，默认使用 base_path
        if not input_path:
            print(f"使用默认路径: {base_path}")
            return base_path

        # 检测操作系统
        current_system = platform.system()
        if current_system in ["Darwin", "Linux"]:  # macOS 和 Linux
            # 去掉路径中的转义符（macOS/Linux 终端拖入文件可能会自动加 `\`）
            input_path = input_path.replace("\\", "")

        # 标准化路径：处理多余分隔符，统一格式
        full_path = os.path.normpath(input_path if os.path.isabs(input_path) else os.path.join(base_path, input_path))

        # 检查路径是否有效
        if os.path.exists(full_path):
            print(f"成功加载路径: {full_path}")
            return full_path
        
        print(f"无效路径: {full_path}")

input_path = get_input_path()

# 自动生成模型路径
def get_model_path(base_path="./models", prefix="swin_insect_classifier_", extension=".pth"):
    while True:
        model_index = input("请输入模型序号(例如 0 对应 swin_insect_classifier_0.pth):").strip()
        model_path = os.path.join(base_path, f"{prefix}{model_index}{extension}")
        if os.path.isfile(model_path):
            print(f"成功加载模型路径: {model_path}")
            return model_path
        print(f"模型文件不存在: {model_path}，请重新输入。")

model_path = get_model_path()
model, classes = load_model_and_classes(model_path)

# 进行预测
predict(input_path, model, transform, classes, device)
