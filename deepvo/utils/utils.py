from PIL import Image


def load_image(path):
    return Image.open(path).convert("RGB")


def Log(tag, message):
    print(f'[{tag}] {message}')
