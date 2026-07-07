# Watermark Forgery Attack - TML 2026

**Team:** atml_team018

## Approach 1
Residual-based copy attack: extract watermark signal from 25 source images by averaging denoised residuals, then inject onto clean targets.

## Final Approach

Use a texture mask plus Kutter Watermark Copy Attack, then evaluate locally to estimate LPIPS score.

## How to Reproduce

1. Download dataset:
hf download SprintML/tml2026_task4 --repo-type dataset --local-dir .
unzip Dataset.zip

2. Install dependencies:
pip install numpy pillow scipy torch torchvision lpips

3. Run pipeline:
python3 kutter.py

4. Evaluate:
python3 local_evaluation.py

6. Submit file:
python3 submission_kutter.py
