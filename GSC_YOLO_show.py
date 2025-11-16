import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

# ==================== GhostConv模块 ====================
class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=None):
        super(GhostConv, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size//2 if padding is None else padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, 
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

# ==================== SimAM注意力机制 ====================
class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

# ==================== GhostBottleneck模块 ====================
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        
        self.conv1 = GhostConv(in_channels, hidden_channels, kernel_size=1)
        
        if self.stride > 1:
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                                  kernel_size//2, groups=hidden_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:
            self.conv2 = None
            
        self.conv3 = GhostConv(hidden_channels, out_channels, kernel_size=1, ratio=2)
        
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        
        if self.conv2 is not None:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act(out)
        
        out = self.conv3(out)
        out = out + self.shortcut(residual)
        out = self.act(out)
        return out

# ==================== C3Ghost模块 ====================
class C3Ghost(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super(C3Ghost, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = GhostConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = GhostConv(in_channels, hidden_channels, 1, 1)
        self.m = nn.Sequential(*[
            GhostBottleneck(hidden_channels, hidden_channels, hidden_channels, 3, 1) 
            for _ in range(n)
        ])
        self.conv3 = GhostConv(2 * hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.m(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv3(x)

# ==================== 改进的NMS算法 ====================
class Cluster_SPM_Dist_NMS:
    def __init__(self, iou_threshold=0.5, sigma=0.2, beta=0.6):
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.beta = beta
    
    def __call__(self, boxes, scores):
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long)
        
        _, order = scores.sort(0, descending=True)
        boxes = boxes[order]
        scores = scores[order]
        
        iou_matrix = self.calculate_iou_matrix(boxes)
        keep = self.cluster_nms(iou_matrix, scores)
        return order[keep]
    
    def calculate_iou_matrix(self, boxes):
        n = boxes.size(0)
        iou_matrix = torch.zeros((n, n), device=boxes.device)
        
        for i in range(n):
            box1 = boxes[i].unsqueeze(0)
            box2 = boxes[i:]
            iou = self.calculate_iou(box1, box2)
            iou_matrix[i, i:] = iou.squeeze(0)
            iou_matrix[i:, i] = iou.squeeze(0)
        
        return iou_matrix
    
    def calculate_iou(self, box1, box2):
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def cluster_nms(self, iou_matrix, scores):
        n = iou_matrix.size(0)
        triu_mask = torch.triu(torch.ones(n, n, device=iou_matrix.device), diagonal=1)
        iou_triu = iou_matrix * triu_mask
        
        keep = torch.ones(n, dtype=torch.bool, device=iou_matrix.device)
        
        for i in range(n):
            if not keep[i]:
                continue
            
            suppress = (iou_triu[i] > self.iou_threshold) & keep
            
            for j in range(i+1, n):
                if suppress[j]:
                    penalty = torch.exp(-(iou_triu[i, j].pow(2) / self.sigma))
                    scores[j] = scores[j] * penalty
                    
                    if scores[j] < 0.01:
                        keep[j] = False
        
        return keep

# ==================== GSC-YOLO模型主体 ====================
class GSC_YOLO(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(GSC_YOLO, self).__init__()
        self.num_classes = num_classes
        
        # 主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 32, 6, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            C3Ghost(32, 64, n=1),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            C3Ghost(128, 128, n=3),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            C3Ghost(256, 256, n=3),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            
            C3Ghost(512, 512, n=1)
        )
        
        # 注意力机制
        self.simam = SimAM()
        
        # 检测头
        self.head = nn.ModuleList([
            nn.Conv2d(128, 3 * (5 + self.num_classes), 1),
            nn.Conv2d(256, 3 * (5 + self.num_classes), 1),
            nn.Conv2d(512, 3 * (5 + self.num_classes), 1)
        ])
        
        # NMS算法
        self.nms = Cluster_SPM_Dist_NMS()
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.simam(features)
        outputs = [head(features) for head in self.head]
        return outputs

# ==================== 图像预处理工具 ====================
class ImageProcessor:
    @staticmethod
    def load_image(image_path, target_size=640):
        """加载并预处理图像"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 调整大小保持宽高比
        image_resized = ImageProcessor.resize_image(image, target_size)
        
        # 转换为tensor
        image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_size, image_resized.size
    
    @staticmethod
    def resize_image(image, target_size):
        """调整图像大小，保持宽高比"""
        w, h = image.size
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)
        
        # 填充到目标尺寸
        new_image = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        new_image.paste(image_resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        
        return new_image

# ==================== 检测结果可视化工具 ====================
class DetectionVisualizer:
    def __init__(self, class_names=None):
        self.class_names = class_names or [f'class_{i}' for i in range(10)]
        self.colors = self.generate_colors(len(self.class_names))
    
    @staticmethod
    def generate_colors(n):
        """生成不同颜色用于不同类别"""
        colors = []
        for i in range(n):
            r = int((i * 123 + 45) % 255)
            g = int((i * 67 + 123) % 255)
            b = int((i * 234 + 67) % 255)
            colors.append((r, g, b))
        return colors
    
    def draw_detections(self, image, boxes, scores, labels, original_size, resized_size):
        """在图像上绘制检测结果"""
        draw = ImageDraw.Draw(image)
        
        # 计算缩放比例
        w_ratio = original_size[0] / resized_size[0]
        h_ratio = original_size[1] / resized_size[1]
        
        for box, score, label in zip(boxes, scores, labels):
            # 转换坐标到原始图像尺寸
            x1 = int(box[0] * w_ratio)
            y1 = int(box[1] * h_ratio)
            x2 = int(box[2] * w_ratio)
            y2 = int(box[3] * h_ratio)
            
            # 绘制边界框
            color = self.colors[label % len(self.colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签和置信度
            label_text = f'{self.class_names[label]}: {score:.2f}'
            draw.rectangle([x1, y1-20, x1+len(label_text)*8, y1], fill=color)
            draw.text((x1+5, y1-18), label_text, fill=(255, 255, 255))
        
        return image

# ==================== 主调用示例 ====================
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化模型
    model = GSC_YOLO(num_classes=10).to(device)
    model.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    
    # 初始化工具类
    processor = ImageProcessor()
    visualizer = DetectionVisualizer()
    
    # 模拟输入图像（在实际使用中替换为真实图像路径）
    # image_path = "your_image.jpg"
    
    # 创建模拟图像用于演示
    demo_image = create_demo_image()
    demo_image_path = "demo_image.jpg"
    demo_image.save(demo_image_path)
    
    try:
        # 加载和预处理图像
        input_tensor, original_size, resized_size = processor.load_image(demo_image_path)
        input_tensor = input_tensor.to(device)
        
        print(f'输入图像尺寸: {input_tensor.shape}')
        
        # 模型推理
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_tensor)
            inference_time = time.time() - start_time
        
        print(f'推理时间: {inference_time:.4f}秒')
        print(f'输出特征图数量: {len(outputs)}')
        
        for i, output in enumerate(outputs):
            print(f'输出 {i} 形状: {output.shape}')
        
        # 模拟检测结果（实际应用中需要实现完整的后处理）
        print("\n模拟检测结果演示:")
        demo_boxes, demo_scores, demo_labels = simulate_detection_results()
        
        # 可视化结果
        result_image = visualizer.draw_detections(
            demo_image, demo_boxes, demo_scores, demo_labels, 
            original_size, resized_size
        )
        
        # 保存结果
        result_image.save("detection_result.jpg")
        print("检测结果已保存为 'detection_result.jpg'")
        
        # 性能统计
        print(f"\n性能统计:")
        print(f"- 模型参数量: {total_params:,}")
        print(f"- 推理时间: {inference_time:.4f}秒")
        print(f"- 检测目标数: {len(demo_boxes)}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        # 使用模拟数据进行演示
        demo_boxes, demo_scores, demo_labels = simulate_detection_results()
        print(f"使用模拟数据演示，检测到 {len(demo_boxes)} 个目标")

def create_demo_image():
    """创建演示图像"""
    image = Image.new('RGB', (640, 480), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    
    # 绘制一些模拟目标
    draw.rectangle([100, 100, 200, 180], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
    draw.rectangle([300, 150, 400, 250], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
    draw.rectangle([200, 300, 300, 380], fill=(0, 0, 255), outline=(0, 0, 0), width=2)
    
    return image

def simulate_detection_results():
    """模拟检测结果用于演示"""
    boxes = [
        [95, 95, 205, 185],   # 调整边界框位置
        [295, 145, 405, 255],
        [195, 295, 305, 385]
    ]
    scores = [0.89, 0.76, 0.92]
    labels = [0, 1, 2]
    
    return boxes, scores, labels

# ==================== 批量处理示例 ====================
class BatchProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.visualizer = DetectionVisualizer()
    
    def process_batch(self, image_paths, batch_size=4):
        """批量处理图像"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = self.process_single_batch(batch_paths)
            results.extend(batch_results)
            
            print(f'已处理 {min(i+batch_size, len(image_paths))}/{len(image_paths)} 张图像')
        
        return results
    
    def process_single_batch(self, image_paths):
        """处理单个批次"""
        batch_tensors = []
        image_infos = []
        
        for path in image_paths:
            try:
                tensor, original_size, resized_size = ImageProcessor.load_image(path)
                batch_tensors.append(tensor)
                image_infos.append((original_size, resized_size, path))
            except Exception as e:
                print(f"加载图像 {path} 失败: {e}")
                continue
        
        if not batch_tensors:
            return []
        
        batch_tensor = torch.cat(batch_tensors, 0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # 处理每个图像的输出
        batch_results = []
        for i, (original_size, resized_size, path) in enumerate(image_infos):
            # 这里简化处理，实际需要完整的后处理
            result = {
                'image_path': path,
                'original_size': original_size,
                'detections': simulate_detection_results()  # 模拟结果
            }
            batch_results.append(result)
        
        return batch_results

if __name__ == "__main__":
    print("GSC-YOLO模型调用示例")
    print("=" * 50)
    main()
