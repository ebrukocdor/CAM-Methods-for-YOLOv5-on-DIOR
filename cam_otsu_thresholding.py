
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError

def Hist(img):
    row, col = img.shape
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1
    return y

def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i] > 0:
            cnt += h[i]
    return cnt

def weight(s, e, h):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w

def mean(s, e, h):
    w = weight(s, e, h)
    if w == 0:
        return 0
    m = 0
    for i in range(s, e):
        m += h[i] * i
    return m / float(w)

def variance(s, e, h):
    w = weight(s, e, h)
    if w == 0:
        return 0
    m = mean(s, e, h)
    v = 0
    for i in range(s, e):
        v += ((i - m) ** 2) * h[i]
    v /= w
    return v

def threshold(h):
    threshold_values = {}
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i, h)
        wb = weight(0, i, h) / float(cnt)
        mb = mean(0, i, h)

        vf = variance(i, len(h), h)
        wf = weight(i, len(h), h) / float(cnt)
        mf = mean(i, len(h), h)

        V2w = wb * vb + wf * vf
        V2b = wb * wf * (mb - mf) ** 2

        if not math.isnan(V2w):
            threshold_values[i] = V2w
    return threshold_values

def get_optimal_threshold(threshold_values):
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    return optimal_threshold[0]

def process_images(input_folder):
    combined_hist = np.zeros(256)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    image = Image.open(image_path).convert("L")
                    img = np.asarray(image)
                    combined_hist += Hist(img)
                except UnidentifiedImageError:
                    print(f"Skipping corrupt or unidentified image: {image_path}")

    threshold_values = threshold(combined_hist)
    op_thres = get_optimal_threshold(threshold_values)

    return op_thres, combined_hist

input_folder = '/content/drive/MyDrive/cam_otsu'

average_threshold, combined_hist = process_images(input_folder)
print(f"Ortalama Eşik Değeri (0-255): {average_threshold}")

normalized_threshold = average_threshold / 255.0
print(f"Normalize Edilmiş Eşik Değeri (0-1): {normalized_threshold}")

