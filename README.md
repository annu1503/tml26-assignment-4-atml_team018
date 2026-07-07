# Watermark Forgery Attack - TML 2026

**Team:** atml_team018

## Approach
Residual-based copy attack: extract watermark signal from 25 source images by averaging denoised residuals, then inject onto clean targets.

## How to Reproduce

1. Download dataset:
hf download SprintML/tml2026_task4 --repo-type dataset --local-dir .
unzip Dataset.zip

2. Install dependencies:
pip install numpy pillow scipy torch torchvision

3. Run pipeline:
python3 forge_watermark.py

4. Submit:
python3 submission.py
