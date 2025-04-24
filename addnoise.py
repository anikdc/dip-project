import os, cv2, numpy as np
from tqdm import tqdm

gauss_sigmas = [15, 25, 30, 50, 70]
sp_amounts   = [0.01, 0.1, 0.2, 0.3]

for s in gauss_sigmas:
    os.makedirs(os.path.join("GaussianNoise", f"noisy_sigma{s}"), exist_ok=True)
for amt in sp_amounts:
    os.makedirs(os.path.join("SaltPepperNoise", f"noisy_sp{int(amt*100)}"), exist_ok=True)

def add_gaussian(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_saltpepper(img, amount):
    out = img.copy()
    n = int(amount * img.size)
    coords = np.random.choice(img.size, n, replace=False)
    out.flat[coords[: n // 2]] = 255
    out.flat[coords[n // 2 :]] = 0
    return out

image_files = [f for f in os.listdir("Clean") if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for name in tqdm(image_files, desc="Processing images"):
    img = cv2.imread(os.path.join("Clean", name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    for sigma in gauss_sigmas:
        cv2.imwrite(os.path.join("GaussianNoise", f"noisy_sigma{sigma}", name), add_gaussian(img, sigma))
    for amt in sp_amounts:
        cv2.imwrite(os.path.join("SaltPepperNoise", f"noisy_sp{int(amt*100)}", name), add_saltpepper(img, amt))
