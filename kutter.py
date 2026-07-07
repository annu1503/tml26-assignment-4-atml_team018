import os
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image


ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("Dataset")
TEMP_OUT_DIR = Path("submission_temp")
FILE_PATH = "submission_kutter.zip"

# Hyperparameters
FLAT_ALPHA = 0.05    
TEXTURE_ALPHA = 50  # start lower since the kutter residual is very concentrated
BLUR_KERNEL = 7      # Size of the Gaussian blur (odd)
BLUR_SIGMA = 2.0     # strength of the blur

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running Kutter Attack on: {DEVICE}")

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# functions
def get_perceptual_mask(images, device):
    """Computes a normalized spatial perceptual mask using Sobel edge detection."""
    grayscale = images.mean(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1.,  0.,  1.], [-2.,  0.,  2.], [-1.,  0.,  1.]]).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1., -2., -1.], [ 0.,  0.,  0.], [ 1.,  2.,  1.]]).view(1, 1, 3, 3).to(device)
    
    G_x = F.conv2d(grayscale, sobel_x, padding=1)
    G_y = F.conv2d(grayscale, sobel_y, padding=1)
    
    magnitude = torch.sqrt(G_x**2 + G_y**2 + 1e-6)
    
    B = magnitude.shape[0]
    mag_flat = magnitude.view(B, -1)
    mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    return (magnitude - mag_min) / (mag_max - mag_min + 1e-6)

def load_image_batch(paths):
    tensors = [to_tensor(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors).to(DEVICE)


if not DATASET_DIR.exists():
    print(f"Unzipping {ZIP_FILE}...")
    DATASET_DIR.mkdir(exist_ok=True) 
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR) 

TEMP_OUT_DIR.mkdir(exist_ok=True)
target_dir = DATASET_DIR / "clean_targets"

# forging loop
print("Extracting and forging via Kutter Method...")

CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200),
]

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} -> Forging targets {target_start} to {target_stop} ...")

    # watermarked sources
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_paths = sorted(list(source_dir.glob("*.png")))
    if not source_paths: continue
    source_batch = load_image_batch(source_paths)

    
    ### kutter method
    # apply gaussian blur to watermarked image
    blurred_sources = TF.gaussian_blur(source_batch, kernel_size=[BLUR_KERNEL, BLUR_KERNEL], sigma=[BLUR_SIGMA, BLUR_SIGMA])
    # substract blur from image to get watermark
    noise_residuals = source_batch - blurred_sources
    # take median to eliminate any features similar to watermark
    kutter_residual = noise_residuals.median(dim=0, keepdim=True).values
    

    # targets
    target_paths = [target_dir / f"{i}.png" for i in range(target_start, target_stop + 1)]
    target_batch = load_image_batch(target_paths)
    
    # inject watermark
    mask = get_perceptual_mask(target_batch, DEVICE)
    margin_mask = 1.0 - torch.abs((target_batch * 2.0) - 1.0)
    final_mask = mask * margin_mask
    
    dynamic_alpha = FLAT_ALPHA + (final_mask * (TEXTURE_ALPHA - FLAT_ALPHA))
    forged_batch = target_batch + (kutter_residual * dynamic_alpha)
    forged_batch = torch.clamp(forged_batch, 0.0, 1.0)
    
    # save outputs
    for i, forged_tensor in enumerate(forged_batch):
        out_path = TEMP_OUT_DIR / target_paths[i].name
        to_pil(forged_tensor.cpu()).save(out_path)
        total_processed += 1

# submission
print(f"ziping {total_processed} images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)
print("Done!")