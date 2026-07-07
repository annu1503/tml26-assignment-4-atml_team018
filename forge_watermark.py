import os
import zipfile
import numpy as np
from PIL import Image
from scipy import ndimage

WATERMARKED_SOURCES_DIR = "./watermarked_sources"
CLEAN_TARGETS_DIR = "./clean_targets"
OUTPUT_DIR = "./forged_outputs"
SUBMISSION_ZIP = "./submission.zip"

SCHEME_TO_RANGE = {
    "WM_1": (1, 25),
    "WM_2": (26, 50),
    "WM_3": (51, 75),
    "WM_4": (76, 100),
    "WM_5": (101, 125),
    "WM_6": (126, 150),
    "WM_7": (151, 175),
    "WM_8": (176, 200),
}

DEFAULT_ALPHA = 1.0
ALPHA_PER_SCHEME = {k: DEFAULT_ALPHA for k in SCHEME_TO_RANGE}


def denoise(img: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        out[..., c] = ndimage.median_filter(img[..., c], size=3)
    return out


def extract_residual(img_path: str) -> np.ndarray:
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    den = denoise(img)
    return img - den


def estimate_watermark_delta(scheme_dir: str) -> np.ndarray:
    files = sorted(
        f for f in os.listdir(scheme_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {scheme_dir}")

    residuals = []
    for f in files:
        residuals.append(extract_residual(os.path.join(scheme_dir, f)))

    delta = np.mean(np.stack(residuals, axis=0), axis=0)
    return delta


def apply_watermark(target_path: str, delta: np.ndarray, alpha: float) -> Image.Image:
    img = np.array(Image.open(target_path).convert("RGB"), dtype=np.float32)

    if delta.shape[:2] != img.shape[:2]:
        delta_img = Image.fromarray(
            np.clip(delta + 128, 0, 255).astype(np.uint8)
        ).resize((img.shape[1], img.shape[0]), Image.BICUBIC)
        delta = np.array(delta_img, dtype=np.float32) - 128.0

    forged = img + alpha * delta
    forged = np.clip(forged, 0, 255).astype(np.uint8)
    return Image.fromarray(forged)


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for scheme, (lo, hi) in SCHEME_TO_RANGE.items():
        scheme_dir = os.path.join(WATERMARKED_SOURCES_DIR, scheme)
        print(f"[{scheme}] estimating watermark delta from {scheme_dir} ...")
        delta = estimate_watermark_delta(scheme_dir)
        alpha = ALPHA_PER_SCHEME[scheme]

        print(f"[{scheme}] delta stats: mean={delta.mean():.4f} "
              f"std={delta.std():.4f} max_abs={np.abs(delta).max():.4f}")

        for idx in range(lo, hi + 1):
            fname = f"{idx}.png"
            target_path = os.path.join(CLEAN_TARGETS_DIR, fname)
            if not os.path.exists(target_path):
                print(f"  WARNING: missing target {target_path}, skipping")
                continue
            forged = apply_watermark(target_path, delta, alpha)
            forged.save(os.path.join(OUTPUT_DIR, fname))

        print(f"[{scheme}] done: images {lo}.png to {hi}.png forged.\n")

    print("All schemes processed.")


def make_submission_zip():
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in range(1, 201):
            fname = f"{idx}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)
            else:
                print(f"WARNING: {fname} missing from output, not included")
    print(f"Wrote {SUBMISSION_ZIP}")


if __name__ == "__main__":
    run_pipeline()
    make_submission_zip()
