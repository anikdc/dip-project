import os, cv2, numpy as np
from tqdm import tqdm

S_MAX=7

def ensure(p):
    os.makedirs(p,exist_ok=True)

def adaptive_median(img,s_max=S_MAX):
    if s_max%2==0 or s_max<3:
        raise ValueError
    pad=s_max//2
    padded=np.pad(img,pad,mode="reflect")
    out=np.empty_like(img)
    rows,cols=img.shape
    for r in range(rows):
        for c in range(cols):
            k=3
            while True:
                half=k//2
                win=padded[r+pad-half:r+pad+half+1,c+pad-half:c+pad+half+1]
                z_min,z_max=win.min(),win.max()
                z_med=np.median(win)
                if z_med>z_min and z_med<z_max:
                    z_xy=padded[r+pad,c+pad]
                    out[r,c]=z_xy if (z_xy>z_min and z_xy<z_max) else z_med
                    break
                k+=2
                if k>s_max:
                    out[r,c]=z_med
                    break
    return out

def process(src,dst):
    img=cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    ensure(os.path.dirname(dst))
    cv2.imwrite(dst,adaptive_median(img))

def run():
    sp_root="SaltPepperNoise"
    if not os.path.isdir(sp_root):
        return
    for dir_name in os.listdir(sp_root):
        src_dir=os.path.join(sp_root,dir_name)
        if not os.path.isdir(src_dir) or not dir_name.startswith("noisy_sp"):
            continue
        label=dir_name.replace("noisy_","")
        dst_dir=os.path.join("Denoised","SaltPepper","Adaptive Median",f"denoised_{label}")
        ensure(dst_dir)
        for name in tqdm(os.listdir(src_dir),desc=f"amf {label}"):
            src=os.path.join(src_dir,name)
            if os.path.isdir(src):
                continue
            dst=os.path.join(dst_dir,name)
            process(src,dst)

if __name__=="__main__":
    run()
