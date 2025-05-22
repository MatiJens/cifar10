import numpy as np
from PIL import Image

def process_single_image(path, label):

    img = Image.open(path).convert("RGB")
    np_img = np.array(img).flatten()

    return np_img, label