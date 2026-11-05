"""app.ico 만들기 (빌드 전에 한 번). 파란 종이 위 'A→가'."""
from PIL import Image, ImageDraw, ImageFont

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((20, 8, 236, 248), radius=28, fill=(47, 111, 222))
d.polygon([(186, 8), (236, 58), (186, 58)], fill=(160, 192, 245))  # 접힌 귀퉁이
bold = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 150)
small = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 70)
d.text((44, 40), "A", font=small, fill=(255, 255, 255, 200))
d.text((128, 150), "가", font=bold, fill="white", anchor="mm")
img.save("app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("app.ico")
