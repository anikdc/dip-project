import os, cv2, numpy as np
from tqdm import tqdm

def ensure(p):
    os.makedirs(p, exist_ok=True)

def fast_median(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    rows, cols = img.shape
    out = np.empty_like(img)
    w = k * k
    half = w // 2
    for r in range(rows):
        hist = np.zeros(256, dtype=np.int32)
        for dy in range(k):
            for dx in range(k):
                v = padded[r + dy, dx]
                hist[v] += 1
        csum = np.cumsum(hist)
        mdn = int(np.searchsorted(csum, half + 1))
        ltmdn = csum[mdn - 1] if mdn > 0 else 0
        out[r, 0] = mdn
        for c in range(1, cols):
            cl = c - 1
            for dy in range(k):
                v = padded[r + dy, cl]
                hist[v] -= 1
                if v < mdn:
                    ltmdn -= 1
            cr = c + k - 1
            for dy in range(k):
                v = padded[r + dy, cr]
                hist[v] += 1
                if v < mdn:
                    ltmdn += 1
            while ltmdn > half:
                mdn -= 1
                ltmdn -= hist[mdn]
            while ltmdn + hist[mdn] <= half:
                ltmdn += hist[mdn]
                mdn += 1
            out[r, c] = mdn
    return out

def process(src, dst, k=3):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    ensure(os.path.dirname(dst))
    cv2.imwrite(dst, fast_median(img, k))

def run():
    root = "SaltPepperNoise"
    if not os.path.isdir(root):
        return
    for d in os.listdir(root):
        src_dir = os.path.join(root, d)
        if not os.path.isdir(src_dir) or not d.startswith("noisy_sp"):
            continue
        label = d.replace("noisy_", "")
        dst_dir = os.path.join("Denoised", "SaltPepper", "Median", f"denoised_{label}")
        ensure(dst_dir)
        for name in tqdm(os.listdir(src_dir), desc=f"median {label}"):
            src = os.path.join(src_dir, name)
            if os.path.isdir(src):
                continue
            dst = os.path.join(dst_dir, name)
            process(src, dst, 3)

if __name__ == "__main__":
    run()
