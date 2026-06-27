from captcha.image import ImageCaptcha

img = ImageCaptcha(width=160, height=60)



tmp = img.generate_image("aut9")

tmp.save("dataset/test/test.png")