from pathlib import Path
import cv2, pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

ROOT=Path(".")
CLEAN_DIR=ROOT/"Clean"
IMG_EXTS={".png",".jpg",".jpeg"}

rows=[]
den_root=ROOT/"Denoised"

for noise in ["Gaussian","SaltPepper"]:
    noise_dir=den_root/noise
    if not noise_dir.is_dir():
        continue
    for method_dir in noise_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method=method_dir.name.replace(" ","_").lower()
        for param_dir in method_dir.iterdir():
            if not param_dir.is_dir() or not param_dir.name.startswith("denoised_"):
                continue
            param=param_dir.name.replace("denoised_","")
            for img_file in param_dir.iterdir():
                if img_file.suffix.lower() not in IMG_EXTS:
                    continue
                ref_path=CLEAN_DIR/img_file.name
                if not ref_path.exists():
                    continue
                ref=cv2.imread(str(ref_path),cv2.IMREAD_GRAYSCALE)
                den=cv2.imread(str(img_file),cv2.IMREAD_GRAYSCALE)
                if ref is None or den is None:
                    continue
                rows.append(dict(
                    image=img_file.stem,
                    param=param,
                    method=method,
                    noise=noise,
                    PSNR_dB=psnr(ref,den,data_range=255),
                    SSIM=ssim(ref,den,data_range=255,channel_axis=None)
                ))

df=pd.DataFrame(rows)
df["PSNR_dB"]=df["PSNR_dB"].round(2)
df["SSIM"]=df["SSIM"].round(4)
df.to_csv("results_detailed.csv",index=False)

mean_df=(df.groupby(["noise","param","method"])
           .agg(Avg_PSNR_dB=("PSNR_dB","mean"),Avg_SSIM=("SSIM","mean"))
           .reset_index()
           .sort_values(["noise","param","Avg_PSNR_dB"],ascending=[True,True,False]))
mean_df.to_csv("results_summary.csv",index=False)

print(df)
print("\n=== Averages ===\n")
print(mean_df.to_string(index=False))
