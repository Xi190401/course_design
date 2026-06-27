import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from model import CaptchaCNN
import sys

sys.path.insert(0, os.path.dirname(__file__))

#数据集类  便于加载数据
class CaptchaDataset(Dataset):
    def __init__(self, folder):
        import os
        self.files = [f for f in os.listdir(folder) if f.endswith('.png')]
        self.folder = folder
        self.chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # 36

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        # 读图，灰度
        img = cv2.imread(f'{self.folder}/{f}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (160, 60))
        img = torch.FloatTensor(img) / 255.0  # 归一化
        img = img.unsqueeze(0)  # (1, 60, 160)

        # 解析标签
        label_str = f.split('_')[1].split('.')[0]  # "AB3D"
        label = [self.chars.index(c) for c in label_str]  # [10, 11, 3, 13]
        label = torch.LongTensor(label)  # (4,)
        return img, label


#参数
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001



#数据加载
train_dataset = CaptchaDataset('dataset/train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"训练图片数量: {len(train_dataset)}")
print(f"每批 {BATCH_SIZE} 张，共 {len(train_loader)} 批")




model = CaptchaCNN(num_classes=36, captcha_length=4).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



#训练循环
# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)   # (batch, 4, 36)

        # 合并所有位置一起算损失
        outputs_flat = outputs.view(-1, 36)   # (batch*4, 36)
        labels_flat = labels.view(-1)          # (batch*4,)
        loss = criterion(outputs_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 统计整张图准确率（用原始维度的 outputs）
        pred = outputs.argmax(dim=2)                    # (batch, 4)
        correct += (pred == labels).all(dim=1).sum().item()
        total += imgs.size(0)

    acc = correct / total
    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.3f}')

#保存模型
torch.save(model.state_dict(), 'models/model1.pth')
print("\n训练完成，模型已保存为 models/model1.pth")