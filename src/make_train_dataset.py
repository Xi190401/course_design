import random
import string
from captcha.image import ImageCaptcha
import os

#创建目录
os.makedirs("dataset/train",exist_ok=True)

img_generator = ImageCaptcha()

# 生成用：包含大小写（让图片多样性更丰富）
CHARS = string.digits + string.ascii_letters   # 62 种


for i in range(20000):

    #随机字符
    text = ''.join(random.choices(CHARS, k=4))

    #生成图片
    tmp = img_generator.generate_image(text)

    #标签转大写
    label = text.upper()

    #保存图片
    tmp.save(f"dataset/train/{i:05d}_{label}.png")



