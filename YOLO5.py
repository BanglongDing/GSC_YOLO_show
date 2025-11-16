import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import math

# ==================== GhostConv模块 ====================
class GhostConv(nn.Module):
    """
    GhostConv模块实现
    通过少量常规卷积和廉价操作生成特征图，减少参数量
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=None):
        super(GhostConv, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        # 主要卷积层
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size//2 if padding is None else padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True)
        )
        
        # 廉价操作 - 深度分离卷积
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, 
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

# ==================== SimAM注意力机制 ====================
class SimAM(nn.Module):
    """
    简单注意力机制模块，基于特征图的统计特性进行自适应加权
    不引入额外参数，计算效率高
    """
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
    """
    GhostBottleneck结构，包含两个GhostConv层和shortcut连接
    支持不同步长的配置
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        
        # 第一个GhostConv扩展通道数
        self.conv1 = GhostConv(in_channels, hidden_channels, kernel_size=1)
        
        # 深度分离卷积（步长为2时使用）
        if self.stride > 1:
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                                  kernel_size//2, groups=hidden_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:
            self.conv2 = None
            
        # 第二个GhostConv还原通道数
        self.conv3 = GhostConv(hidden_channels, out_channels, kernel_size=1, ratio=2)
        
        # Shortcut连接
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 激活函数
        self.act = nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        # 第一个GhostConv
        out = self.conv1(x)
        out = self.act(out)
        
        # 深度分离卷积（如果需要）
        if self.conv2 is not None:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act(out)
        
        # 第二个GhostConv
        out = self.conv3(out)
        
        # Shortcut连接
        out = out + self.shortcut(residual)
        out = self.act(out)
        
        return out

# ==================== C3Ghost模块 ====================
class C3Ghost(nn.Module):
    """
    改进的C3模块，使用GhostBottleneck替代标准Bottleneck
    减少参数量同时保持特征提取能力
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super(C3Ghost, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        # 第一个卷积层
        self.conv1 = GhostConv(in_channels, hidden_channels, 1, 1)
        
        # 第二个卷积层
        self.conv2 = GhostConv(in_channels, hidden_channels, 1, 1)
        
        # 多个GhostBottleneck
        self.m = nn.Sequential(*[
            GhostBottleneck(hidden_channels, hidden_channels, hidden_channels, 3, 1) 
            for _ in range(n)
        ])
        
        # 输出卷积层
        self.conv3 = GhostConv(2 * hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        # 两条路径
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # 主干路径通过多个GhostBottleneck
        x1 = self.m(x1)
        
        # 合并两条路径
        x = torch.cat((x1, x2), dim=1)
        
        return self.conv3(x)

# ==================== 改进的NMS算法 ====================
class Cluster_SPM_Dist_NMS:
    """
    改进的非极大值抑制算法
    包含Cluster_NMS、得分惩罚机制和中心点距离机制
    """
    def __init__(self, iou_threshold=0.5, sigma=0.2, beta=0.6):
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.beta = beta
    
    def __call__(self, boxes, scores):
        """
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N]
        """
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long)
        
        # 按得分排序
        _, order = scores.sort(0, descending=True)
        boxes = boxes[order]
        scores = scores[order]
        
        # 计算IoU矩阵
        iou_matrix = self.calculate_iou_matrix(boxes)
        
        # Cluster_NMS算法
        keep = self.cluster_nms(iou_matrix, scores)
        
        return order[keep]
    
    def calculate_iou_matrix(self, boxes):
        """计算IoU矩阵"""
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
        """计算两个边界框的IoU"""
        # 计算交集
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def calculate_diou(self, box1, box2):
        """计算DIoU（距离IoU）"""
        iou = self.calculate_iou(box1, box2)
        
        # 计算中心点距离
        center1_x = (box1[:, 0] + box1[:, 2]) / 2
        center1_y = (box1[:, 1] + box1[:, 3]) / 2
        center2_x = (box2[:, 0] + box2[:, 2]) / 2
        center2_y = (box2[:, 1] + box2[:, 3]) / 2
        
        center_distance = (center1_x - center2_x).pow(2) + (center1_y - center2_y).pow(2)
        
        # 计算最小包围矩形的对角线距离
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
        
        enclose_distance = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2)
        
        diou = iou - (center_distance / (enclose_distance + 1e-8))
        
        return diou
    
    def cluster_nms(self, iou_matrix, scores):
        """Cluster_NMS算法实现"""
        n = iou_matrix.size(0)
        
        # 创建上三角矩阵（对角线为0）
        triu_mask = torch.triu(torch.ones(n, n, device=iou_matrix.device), diagonal=1)
        iou_triu = iou_matrix * triu_mask
        
        # 迭代处理
        keep = torch.ones(n, dtype=torch.bool, device=iou_matrix.device)
        
        for i in range(n):
            if not keep[i]:
                continue
            
            # 找到与当前框IoU大于阈值的框
            suppress = (iou_triu[i] > self.iou_threshold) & keep
            
            # 应用得分惩罚机制和中心点距离机制
            for j in range(i+1, n):
                if suppress[j]:
                    # 得分惩罚
                    penalty = torch.exp(-(iou_triu[i, j].pow(2) / self.sigma))
                    
                    # 中心点距离惩罚
                    diou = self.calculate_diou(iou_matrix[i:i+1], iou_matrix[j:j+1])
                    dist_penalty = (diou.pow(2) / self.beta) if self.beta > 0 else 0
                    
                    # 综合惩罚
                    total_penalty = torch.min(penalty + dist_penalty, torch.tensor(1.0))
                    
                    scores[j] = scores[j] * total_penalty
                    
                    # 如果得分过低，则抑制该框
                    if scores[j] < 0.01:
                        keep[j] = False
        
        return keep

# ==================== GSC-YOLO模型主体 ====================
class GSC_YOLO(nn.Module):
    """
    完整的GSC-YOLO模型实现
    基于YOLOv5架构，集成GhostConv、SimAM和改进的NMS
    """
    def __init__(self, num_classes=10, channels=3):
        super(GSC_YOLO, self).__init__()
        self.num_classes = num_classes
        
        # 主干网络 (Backbone)
        self.backbone = self._build_backbone(channels)
        
        # 颈部网络 (Neck) - 特征金字塔
        self.neck = self._build_neck()
        
        # 检测头 (Head)
        self.head = self._build_head()
        
        # 注意力机制
        self.simam = SimAM()
        
        # NMS算法
        self.nms = Cluster_SPM_Dist_NMS()
        
    def _build_backbone(self, channels):
        """构建主干特征提取网络"""
        return nn.Sequential(
            # 初始卷积层
            nn.Conv2d(channels, 32, 6, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            
            # C3Ghost模块序列
            C3Ghost(32, 64, n=1),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            
            C3Ghost(128, 128, n=3),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            
            C3Ghost(256, 256, n=3),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            
            C3Ghost(512, 512, n=1)
        )
    
    def _build_neck(self):
        """构建特征金字塔网络"""
        return nn.Sequential(
            # SPPF模块 (空间金字塔池化)
            nn.Conv2d(512, 256, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            
            # 上采样和特征融合
            nn.Upsample(scale_factor=2, mode='nearest'),
            C3Ghost(512, 256, n=1, shortcut=False),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            C3Ghost(512, 128, n=1, shortcut=False),
            
            # 下采样路径
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            C3Ghost(256, 256, n=1, shortcut=False),
            
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True) if hasattr(nn, 'SiLU') else nn.ReLU(inplace=True),
            C3Ghost(512, 512, n=1, shortcut=False)
        )
    
    def _build_head(self):
        """构建检测头"""
        return nn.ModuleList([
            # 三个尺度的检测头
            nn.Conv2d(128, 3 * (5 + self.num_classes), 1),
            nn.Conv2d(256, 3 * (5 + self.num_classes), 1),
            nn.Conv2d(512, 3 * (5 + self.num_classes), 1)
        ])
    
    def forward(self, x):
        # 主干网络
        features = self.backbone(x)
        
        # 应用注意力机制
        features = self.simam(features)
        
        # 颈部网络
        neck_features = self.neck(features)
        
        # 检测头
        if isinstance(neck_features, (list, tuple)):
            outputs = [head(feat) for head, feat in zip(self.head, neck_features)]
        else:
            outputs = [head(neck_features) for head in self.head]
        
        return outputs
    
    def postprocess(self, predictions, conf_threshold=0.25, iou_threshold=0.5):
        """
        后处理函数，应用改进的NMS算法
        """
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for i, pred in enumerate(predictions):
            # 将预测结果转换为边界框格式
            batch_size, _, grid_h, grid_w = pred.shape
            pred = pred.view(batch_size, 3, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            for b in range(batch_size):
                batch_boxes = []
                batch_scores = []
                batch_labels = []
                
                for anchor in range(3):
                    anchor_pred = pred[b, anchor]
                    
                    # 应用置信度阈值
                    conf_mask = anchor_pred[..., 4] > conf_threshold
                    if not conf_mask.any():
                        continue
                    
                    # 解码边界框
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(grid_h), torch.arange(grid_w), indexing='ij'
                    )
                    grid_x = grid_x.to(pred.device).float()
                    grid_y = grid_y.to(pred.device).float()
                    
                    # 这里简化了边界框解码过程，实际需要根据YOLO的锚点机制实现
                    # 详细实现需要参考YOLOv5的具体解码方式
                    
                    # 应用NMS
                    if len(batch_boxes) > 0:
                        boxes_tensor = torch.cat(batch_boxes)
                        scores_tensor = torch.cat(batch_scores)
                        labels_tensor = torch.cat(batch_labels)
                        
                        keep_indices = self.nms(boxes_tensor, scores_tensor)
                        
                        batch_boxes = [boxes_tensor[keep_indices]]
                        batch_scores = [scores_tensor[keep_indices]]
                        batch_labels = [labels_tensor[keep_indices]]
                
                if len(batch_boxes) > 0:
                    all_boxes.append(torch.cat(batch_boxes))
                    all_scores.append(torch.cat(batch_scores))
                    all_labels.append(torch.cat(batch_labels))
        
        return all_boxes, all_scores, all_labels

# ==================== 训练和评估工具 ====================
class GSC_YOLO_Trainer:
    """
    GSC-YOLO模型训练器
    包含数据加载、训练循环、验证等功能
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_metrics = {'precision': 0, 'recall': 0, 'mAP': 0}
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # 计算评估指标（简化版）
                # 实际实现需要完整的mAP计算逻辑
                
        return total_loss / len(dataloader), {k: v/len(dataloader) for k, v in total_metrics.items()}

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建模型
    model = GSC_YOLO(num_classes=10)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 模拟输入
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # 后处理示例
    boxes, scores, labels = model.postprocess(outputs)
    print(f"Detected {len(boxes[0]) if len(boxes) > 0 else 0} objects")