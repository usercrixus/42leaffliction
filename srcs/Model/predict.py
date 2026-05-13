import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .loader import default_transform
from .model import build_model


def load_image(path: Path, img_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tfm = default_transform(img_size)
        t = tfm(img)
    return t.unsqueeze(0)


def show_prediction_window(
    pred_img_path: Path,
    orig_img_path: Path,
    pred_class: str,
    conf: float,
) -> None:
    """Display the prediction alongside transformed and original images."""
    with Image.open(orig_img_path).convert("RGB") as img_orig:
        np_orig = np.asarray(img_orig)
    with Image.open(pred_img_path).convert("RGB") as img_pred:
        np_pred = np.asarray(img_pred)

    fig = plt.figure(
        num="Leaf Prediction",
        figsize=(9, 7.5),
        facecolor="#222222",
        constrained_layout=True,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 0.8, 1.1])
    gs.update(hspace=0.04)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_title = fig.add_subplot(gs[1, :])
    ax_text = fig.add_subplot(gs[2, :])

    ax_orig.imshow(np_orig)
    ax_pred.imshow(np_pred)
    for ax in (ax_orig, ax_pred):
        ax.axis("off")
        ax.set_facecolor("#222222")

    ax_title.set_facecolor("#222222")
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        "DL classification",
        color="white",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    ax_text.set_facecolor("#222222")
    ax_text.axis("off")
    ax_text.text(
        0.5,
        0.75,
        f"Class predicted : {pred_class}",
        color="#7ed957",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    plt.show(block=True)


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict a class for a single image"
    )
    parser.add_argument(
        "img_stem",
        type=str,
        help="Base path without tail or extension (e.g., data/.../image (2))",
    )
    parser.add_argument(
        "--name_tail",
        type=str,
        default="_original",
        help=(
            "Tail appended to the base path for the transformed image "
            "(e.g., _mask, _blur)"
        ),
    )
    parser.add_argument(
        "--orig_tail",
        type=str,
        default="_original",
        help=(
            "Tail appended to the base path for the original image "
            "(defaults to _original)"
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".JPG",
        help="Image extension (with or without leading dot). Defaults to .JPG",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="Model/checkpoints/best.pt"
    )
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    base_path = Path(args.img_stem)
    pred_path = Path(f"{base_path}{args.name_tail}{ext}")
    orig_img_path = Path(f"{base_path}{args.orig_tail}{ext}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Image not found: {pred_path}")
    if not orig_img_path.exists():
        raise FileNotFoundError(f"Original image not found: {orig_img_path}")
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not pred_path.is_file() or pred_path.suffix not in exts:
        raise ValueError(f"File {pred_path} is not a supported image type")
    if not orig_img_path.is_file() or orig_img_path.suffix not in exts:
        raise ValueError(f"File {orig_img_path} is not a supported image type")

    checkpoint_loaded = torch.load(checkpoint, map_location="cpu")
    class_to_idx = checkpoint_loaded.get("class_to_idx")
    if not class_to_idx:
        raise ValueError("Checkpoint missing 'class_to_idx'")
    model = build_model(len(class_to_idx))
    model.load_state_dict(checkpoint_loaded.get("model_state"), strict=False)
    model.eval()

    x = load_image(pred_path, args.img_size)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_class = idx_to_class[int(pred_idx)]
    print(f"Prediction: {pred_class} (p={float(conf):.4f})")
    show_prediction_window(pred_path, orig_img_path, pred_class, float(conf))


if __name__ == "__main__":
    main()
