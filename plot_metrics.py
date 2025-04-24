import re, pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results_summary.csv")
if "noise" not in df.columns:
    df["noise"] = "Gaussian"

def key(txt):
    nums = re.findall(r"[0-9.]+", txt)
    return float(nums[0]) if nums else txt.lower()

for metric, ylabel in [("Avg_PSNR_dB", "PSNR (dB)"), ("Avg_SSIM", "SSIM")]:
    for noise in df["noise"].unique():
        sub = df[df["noise"] == noise].copy()
        sub["order"] = sub["param"].apply(key)
        sub = sub.sort_values("order")
        wide = sub.pivot(index="param", columns="method", values=metric)
        ax = wide.plot(kind="bar")
        ax.set_title(f"{ylabel} – {noise}")
        ax.set_xlabel("Noise level")
        ax.set_ylabel(ylabel)
        ax.legend(title="Method")
        plt.tight_layout()
        plt.show()
