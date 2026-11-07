from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (700, 200), "black")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype(
    "C:/Windows/Fonts/seguiemj.ttf",
    70
)

draw.text(
    (30, 50),
    "😄  😮  😐  ✋  ✊  👍  ✌️  ☝️",
    font=font,
    fill="white"
)

img.save("emoji_test.png")

print("Emoji image created successfully!")
