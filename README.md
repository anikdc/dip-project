# Digital Image Processing Course Project
This is the code and results mentioned in my project report. The topic is "Comparative Analysis of Image Denoising
Methods". For the complete project report, please check the Google Drive link.
## Install Dependencies
All Python libraries are pinned in **requirements.txt**.  
Create (optional) venv, then run:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Denoise images
```bash
python denoise_bilateral.py
python denoise_nlm.py
python denoise_median.py
python denoise_adaptive.py
```
Each script reads its corresponding *noisy* images, applies the filter, and writes results to `Denoised/` following the folder scheme shown above.

### 2. Compute metrics
```bash
python compute_metrics.py
```

### 3. Plot results
```bash
python plot_metrics.py
```

---

## Output / Results Files

| File | Description |
|------|-------------|
| **GaussianPSNR.png** | PSNR comparison for Bilateral vs NLM across σ ∈ {15, 30, 50, 70}. |
| **GaussianSSIM.png** | SSIM comparison for the same Gaussian experiments. |
| **SaltPepperPSNR.png** | PSNR comparison for Median vs Adaptive Median across 1 %, 10 %, 20 %, 30 % corruption. |
| **SaltPepperSSIM.png** | SSIM comparison for the Salt‑and‑Pepper experiments. |

