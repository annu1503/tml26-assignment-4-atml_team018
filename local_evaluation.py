import math
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision import transforms

TARGETS_DIR = Path("Dataset/clean_targets")
FORGED_DIR = Path("submission_temp")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running evaluation on: {DEVICE}")

# initialize the LPIPS model
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

# scale images to fit LPIPS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# evaluation loop
def evaluate_submission():
    if not FORGED_DIR.exists():
        print("Error: submission_temp folder not found. Run attack script first!")
        return

    total_lpips = 0.0
    num_images = 0

    print("Calculating perceptual distances...")
    
    with torch.no_grad():
        for i in range(1, 201):
            target_path = TARGETS_DIR / f"{i}.png"
            forged_path = FORGED_DIR / f"{i}.png"
            
            if not forged_path.exists():
                print(f"[Warning] Missing forged image: {i}.png")
                continue
                
            # Load and transform images
            target_img = Image.open(target_path).convert("RGB")
            forged_img = Image.open(forged_path).convert("RGB")
            
            # Convert to [-1, 1] tensors, add batch dimension
            target_tensor = transform(target_img).unsqueeze(0).to(DEVICE)
            forged_tensor = transform(forged_img).unsqueeze(0).to(DEVICE)
            
            # Calculate distance
            distance = loss_fn(target_tensor, forged_tensor)
            total_lpips += distance.item()
            num_images += 1

    if num_images == 0:
        print("No images were evaluated.")
        return

    # final scoring
    avg_lpips = total_lpips / num_images
    
    # scoring equation for visual quality: e^(-8 * LPIPS)
    visual_quality_score = math.exp(-8 * avg_lpips)
    
    print("\n" + "="*40)
    print("LOCAL EVALUATION RESULTS")
    print("="*40)
    print(f"Images Scored:   {num_images}/200")
    print(f"Average LPIPS:   {avg_lpips:.5f}  (distance from normal image)")
    print(f"S_qlt Score:     {visual_quality_score:.5f}  (multiplier for final score)")
    print("="*40)

if __name__ == "__main__":
    evaluate_submission()