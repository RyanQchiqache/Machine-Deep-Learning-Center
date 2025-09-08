import os
import json
import argparse
import torch
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)

HF_BASES = {
    "mask2former": "facebook/mask2former-swin-small-ade-semantic",
    "segformer": "nvidia/segformer-b2-finetuned-ade-512-512",
    "upernet": "openmmlab/upernet-swin-small",
}

# Processor defaults (match typical training)
PROC_DEFAULTS = {
    "mask2former": dict(reduce_labels=False, do_rescale=True),
    "segformer": dict(reduce_labels=False, do_rescale=True, size={"height": 512, "width": 512}),
    "upernet": dict(reduce_labels=False, do_rescale=True),
}

def _load_state_dict(pth_path: str):
    sd = torch.load(pth_path, map_location="cpu")
    # unwrap common checkpoint formats
    if isinstance(sd, dict):
        for key in ["state_dict", "model", "model_state_dict", "ema_state_dict"]:
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    # strip DDP prefix if present
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd

def _labels(num_classes: int, labels_path: str = None):
    if labels_path and os.path.isfile(labels_path):
        with open(labels_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) != num_classes:
            raise ValueError(f"labels file has {len(names)} names but num_classes={num_classes}")
        id2label = {i: names[i] for i in range(num_classes)}
    else:
        id2label = {i: f"Class_{i}" for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def convert(model_type: str,
            pth_path: str,
            out_dir: str,
            num_classes: int,
            base: str = None,
            labels_path: str = None,
            ignore_index: int = 255,
            reduce_labels: bool = None,
            do_rescale: bool = None,
            size: str = None):
    """
    model_type: mask2former | segformer | upernet
    pth_path:   path to .pth (PyTorch state_dict)
    out_dir:    folder to write HF model/processor
    base:       optional base hub id or local dir; defaults to HF_BASES[model_type]
    labels_path: optional text file with one class name per line
    ignore_index: label to ignore at loss/postprocess time
    reduce_labels/do_rescale/size: processor overrides (size like "512x512")
    """
    model_type = model_type.lower()
    assert model_type in HF_BASES, f"Unsupported model_type: {model_type}"
    base = base or HF_BASES[model_type]

    # Build labels
    id2label, label2id = _labels(num_classes, labels_path)

    # Build model
    if model_type == "mask2former":
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            base, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )
    elif model_type == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained(
            base, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )
    elif model_type == "upernet":
        model = UperNetForSemanticSegmentation.from_pretrained(
            base, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )
    else:
        raise ValueError("Should not happen")

    model.config.ignore_index = ignore_index

    # Load your .pth weights
    sd = _load_state_dict(pth_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) < 20 and (missing or unexpected):
        print("  missing keys (sample):", missing[:10])
        print("  unexpected keys (sample):", unexpected[:10])

    # Prepare processor (AutoImageProcessor is correct for all three)
    proc_kwargs = PROC_DEFAULTS[model_type].copy()
    if reduce_labels is not None:
        proc_kwargs["reduce_labels"] = bool(reduce_labels)
    if do_rescale is not None:
        proc_kwargs["do_rescale"] = bool(do_rescale)
    if size:
        # parse "HxW"
        if isinstance(size, str) and "x" in size.lower():
            h, w = size.lower().split("x")
            proc_kwargs["size"] = {"height": int(h), "width": int(w)}
        else:
            raise ValueError("size must be like '512x512'")

    proc = AutoImageProcessor.from_pretrained(base, **proc_kwargs)

    # Save everything
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    proc.save_pretrained(out_dir)

    # Optional: also dump a tiny metadata file for your reference
    meta = {
        "model_type": model_type,
        "base": base,
        "num_labels": num_classes,
        "ignore_index": ignore_index,
        "processor_kwargs": proc_kwargs,
    }
    with open(os.path.join(out_dir, "conversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved HF-compatible model & processor to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", required=True, choices=["mask2former", "segformer", "upernet"])
    ap.add_argument("--pth", required=True, help="path to .pth state_dict checkpoint")
    ap.add_argument("--out", required=True, help="output directory for save_pretrained")
    ap.add_argument("--num_classes", required=True, type=int)
    ap.add_argument("--base", default=None, help="hub id or local dir; defaults to architecture base")
    ap.add_argument("--labels", default=None, help="optional path to labels.txt (one name per line)")
    ap.add_argument("--ignore_index", default=255, type=int)
    ap.add_argument("--reduce_labels", default=None, type=lambda x: x.lower() in {"1","true","yes"})
    ap.add_argument("--do_rescale", default=None, type=lambda x: x.lower() in {"1","true","yes"})
    ap.add_argument("--size", default=None, help="e.g., 512x512 (SegFormer)")
    args = ap.parse_args()

    convert(
        model_type=args.model_type,
        pth_path=args.pth,
        out_dir=args.out,
        num_classes=args.num_classes,
        base=args.base,
        labels_path=args.labels,
        ignore_index=args.ignore_index,
        reduce_labels=args.reduce_labels,
        do_rescale=args.do_rescale,
        size=args.size,
    )
"""python convert_hf_pth.py \
  --model_type segformer \
  --pth /path/to/segformer_run.pth \
  --out /path/to/hf_export/segformer_flair_2025-08-25 \
  --num_classes 19 \
  --size 512x512 \
  --reduce_labels false \ # for segformer
  --do_rescale true # for segformer
"""