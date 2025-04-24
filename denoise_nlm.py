import os, cv2
from tqdm import tqdm

SIGMAS=[15,25,30,50,70]

def ensure(p):
    os.makedirs(p,exist_ok=True)

def nlm(img,sigma):
    h=int(0.7*sigma)
    return cv2.fastNlMeansDenoising(img,None,h,7,21)

def process(src,dst,sigma):
    img=cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    ensure(os.path.dirname(dst))
    cv2.imwrite(dst,nlm(img,sigma))

def run():
    for sigma in SIGMAS:
        src_dir=os.path.join("GaussianNoise",f"noisy_sigma{sigma}")
        if not os.path.isdir(src_dir):
            continue
        dst_dir=os.path.join("Denoised","Gaussian","NLM",f"denoised_sigma{sigma}")
        ensure(dst_dir)
        for name in tqdm(os.listdir(src_dir),desc=f"nlm σ={sigma}"):
            src=os.path.join(src_dir,name)
            if os.path.isdir(src):
                continue
            dst=os.path.join(dst_dir,name)
            process(src,dst,sigma)

if __name__=="__main__":
    run()
