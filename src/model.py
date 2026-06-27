import torch.nn as nn


class CaptchaCNN(nn.Module):
    """
    CNN 模型：用于识别 4 位验证码
    输入：1×60×160 的灰度图片
    输出：4×36 的预测概率（4 个位置，每个位置 36 种字符）
    """
    def __init__(self, num_classes=36, captcha_length=4):
        super().__init__()

        # ====== 卷积部分：提取特征，逐步缩小 ======
        self.conv = nn.Sequential(
            # 第 1 组：1 → 32 通道，图片 60×160 → 30×80
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 2 组：32 → 64 通道，30×80 → 15×40
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 3 组：64 → 128 通道，15×40 → 7×20
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 4 组：128 → 256 通道，7×20 → 3×10
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ====== 全连接部分：分类 ======
        self.fc = nn.Sequential(
            nn.Flatten(),                      # 256×3×10 = 7680 个数字
            nn.Linear(256 * 3 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),                   # 训练时随机关闭 50% 神经元，防止死记硬背
            nn.Linear(1024, captcha_length * num_classes),   # 输出 4×36 = 144
        )

    def forward(self, x):
        features = self.conv(x)      # 卷积提取特征
        output = self.fc(features)   # 全连接分类
        # 变形： (batch, 144) → (batch, 4, 36)
        output = output.view(output.size(0), 4, 36)
        return output