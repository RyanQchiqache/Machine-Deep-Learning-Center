# infer_mask2former_overlap.py
import argparse
import numpy as np
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from typing import Tuple

# ---- Your classes & colors ----
COLOR_MAP_dense = {
    0: ['Low vegetation', '#f423e8'],
    1: ['Paved road', '#66669c'],
    2: ['Non paved road', '#be9999'],
    3: ['Paved parking place', '#999999'],
    4: ['Non paved parking place', '#faaa1e'],
    5: ['Bikeways', '#98fb98'],
    6: ['Sidewalks', '#4682b4'],
    7: ['Entrance exit', '#6b8e23'],
    8: ['Danger area', '#dcdc00'],
    9: ['Lane-markings', '#ff0000'],
    10: ['Building', '#dc143c'],
    11: ['Car', '#7d008e'],
    12: ['Trailer', '#aac828'],
    13: ['Van', '#c83c64'],
    14: ['Truck', '#961250'],
    15: ['Long truck', '#51b451'],
    16: ['Bus', '#bef115'],
    17: ['Clutter', '#0b7720'],
    18: ['Impervious surface', '#78f078'],
    19: ['Tree', '#464646'],
}
PALETTE = {k: tuple(int(COLOR_MAP_dense[k][1][i:i+2], 16) for i in (1,3,5)) for k in COLOR_MAP_dense}

def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items(): out[mask == k] = rgb
    return out

def hann2d(side: int, eps: float = 1e-6) -> np.ndarray:
    wy = np.hanning(side); wx = np.hanning(side)
    w = np.outer(wy, wx).astype(np.float32)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    return np.clip(w, eps, 1.0)  # avoid zeros

def pad_to_fit(img: np.ndarray, patch: int) -> Tuple[np.ndarray, Tuple[int,int]]:
    H, W = img.shape[:2]
    pad_h = (patch - (H % patch)) % patch
    pad_w = (patch - (W % patch)) % patch
    if pad_h == 0 and pad_w == 0: return img, (H, W)
    img_pad = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="constant", constant_values=0)
    return img_pad, (H, W)

def grid_coords(H: int, W: int, patch: int, overlap_frac: float):
    assert 0 <= overlap_frac < 1.0
    step = max(1, int(patch * (1.0 - overlap_frac)))
    coords = []
    for y in range(0, max(1, H - patch + 1), step):
        for x in range(0, max(1, W - patch + 1), step):
            coords.append((y, x))
    # force right/bottom edges covered
    for y in range(0, max(1, H - patch + 1), step): coords.append((y, W - patch))
    for x in range(0, max(1, W - patch + 1), step): coords.append((H - patch, x))
    coords.append((H - patch, W - patch))
    # dedup
    seen, uniq = set(), []
    for c in coords:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

@torch.no_grad()
def infer_overlap(
    image_path: str,
    model_dir: str,
    num_classes: int = 20,
    ignore_index: int = 255,
    patch_size: int = 512,
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str = "cuda"
):
    device = torch.device(device)
    # load model/processor
    processor = AutoImageProcessor.from_pretrained(model_dir, reduce_labels=False, do_rescale=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device).eval()
    model.config.ignore_index = ignore_index

    # load image
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    # pad so patches fit; remember original size for final crop
    img_pad, (orig_h, orig_w) = pad_to_fit(img, patch_size)
    H, W = img_pad.shape[:2]
    coords = grid_coords(H, W, patch_size, overlap)
    win = hann2d(patch_size)  # blending weights

    # canvases for seam-aware merge (take label where weight is higher)
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    label_canvas  = np.full((H, W), -1, dtype=np.int32)

    # run in batches
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        batch_imgs = [img_pad[y:y+patch_size, x:x+patch_size, :] for (y, x) in batch_coords]
        enc = processor(images=batch_imgs, return_tensors="pt")
        enc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        out = model(**enc)
        preds = processor.post_process_semantic_segmentation(out, target_sizes=[(patch_size, patch_size)]*len(batch_imgs))

        for (y, x), p in zip(batch_coords, preds):
            p = np.asarray(p, dtype=np.int32)
            wview = weight_canvas[y:y+patch_size, x:x+patch_size]
            lview = label_canvas[y:y+patch_size, x:x+patch_size]
            chooser = win > wview
            if np.any(chooser):
                lview[chooser] = p[chooser]
                wview[chooser] = win[chooser]

    # crop back to original size and fix any holes
    label_canvas = label_canvas[:orig_h, :orig_w]
    if (label_canvas < 0).any(): label_canvas[label_canvas < 0] = 0

    return img, label_canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to large JPG/PNG.")
    parser.add_argument("--model", required=True, help="Path to fine-tuned Mask2Former dir (save_pretrained).")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5, help="Fraction of patch size (e.g., 0.5 => 256px).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-color", default=None, help="Optional path to save colorized PNG.")
    parser.add_argument("--save-mask", default=None, help="Optional path to save raw labels (uint16 PNG).")
    args = parser.parse_args()

    img, pred = infer_overlap(
        image_path=args.image,
        model_dir=args.model,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device
    )

    color = colorize(pred)

    # show
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(img);   plt.title("Original");  plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(color); plt.title("Prediction"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # optional save
    if args.save_color:
        Image.fromarray(color).save(args.save_color)
    if args.save_mask:
        # save labels losslessly as 16-bit
        Image.fromarray(pred.astype(np.uint16)).save(args.save_mask)

if __name__ == "__main__":
    import argparse
    main()
"""
run :
python /home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/small_scripts.py \
  --image /home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/DLR_dataset/SS_Dense/val/images/2012-04-26-Muenchen-Tunnel_4K0G0110.jpg \
  --model /home/ryqc/data/experiments/segmentation/checkpoints/dlr/Mask2former/enc-swin_t_w-ade20k/Mask2former_dlr_2025-08-19_14-05-02.pth \
  --num-classes 20 \
  --patch-size 512 \
  --overlap 0.5 \
  --batch-size 4 \
  --device cuda \
  --save-color pred_color.png \
  --save-mask pred_mask.png

"""