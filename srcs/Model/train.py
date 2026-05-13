import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from .loader import build_loaders
from .model import build_model
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # predict
        # Picks the index of the largest value along dimension 1.
        preds = logits.argmax(1)
        # Counts how many were correct in this batch and adds to correct.
        correct += (preds == y).sum().item()
        # Adds the number of samples in this batch to total.
        total += y.size(0)
    return correct / max(1, total)


def create_model_output_path(out):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def train(train_loader, val_loader, class_to_idx, lr, epochs, out):
    out_path = create_model_output_path(out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        running_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=True,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            total += y.size(0)
            avg_loss = running_loss / max(1, total)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")
        train_loss = running_loss / max(1, total)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                },
                out_path,
            )
            print(f"  Saved checkpoint to {out_path}")
    print(f"Best val_acc: {best_acc:.4f}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a leaf classifier with an 80/20 split"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/images_transformed"
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="Model/checkpoints/best.pt")
    parser.add_argument(
        "--name_tail",
        type=str,
        default="_original",
        help=(
            "Only keep files whose basename ends with this tail; "
            "pass empty string to keep all"
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    name_tail = args.name_tail or None
    train_loader, val_loader, class_to_idx = build_loaders(
        args.data_dir,
        args.img_size,
        args.batch_size,
        0.2,
        args.seed,
        name_tail,
    )
    train(
        train_loader,
        val_loader,
        class_to_idx,
        args.lr,
        args.epochs,
        args.out,
    )


if __name__ == "__main__":
    main()
