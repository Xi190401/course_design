import torch
import cv2
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import CaptchaCNN

CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# 加载模型
model = CaptchaCNN(num_classes=36, captcha_length=4)
model.load_state_dict(torch.load('models/model1.pth', map_location='cpu'))
model.eval()

# 读图
img = cv2.imread('dataset/test/test1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (160, 60))
img = torch.FloatTensor(img) / 255.0
img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 160)

# 预测
with torch.no_grad():
    output = model(img)          # (1, 4, 36)
    pred = output.argmax(dim=2)  # (1, 4)
    result = ''.join(CHARS[i] for i in pred[0].tolist())

print(f"识别结果: {result}")