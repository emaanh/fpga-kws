"""Float training of DS-CNN on Speech Commands v2.

    uv run python -m kws.train              # full run
    uv run python -m kws.train --epochs 1   # smoke test
"""

import argparse
import time

import torch
from torch import nn

from .config import CKPT_DIR, CLASSES
from .data import Split
from .features import LogMel
from .model import DSCNN


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(predict, split, batch_size):
    """`predict` maps a batch of audio to logits (or predicted class indices)."""
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for x, y in split.batches(batch_size):
        out = predict(x)
        pred = out.argmax(1) if out.dim() == 2 else out
        confusion.index_put_((y.cpu(), pred.cpu()), torch.ones_like(y.cpu()), accumulate=True)
    return confusion.diag().sum().item() / confusion.sum().item(), confusion


def print_confusion(confusion):
    names = [c.strip("_")[:5] for c in CLASSES]
    print("true\\pred " + " ".join(f"{n:>5}" for n in names))
    for name, row in zip(names, confusion.tolist()):
        print(f"{name:>9} " + " ".join(f"{v:>5}" for v in row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gain-db", type=float, default=10.0, help="random speech gain range (+-dB)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--out", default="dscnn_float.pt")
    args = p.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    train = Split("train", device, gain_db=args.gain_db)
    val = Split("val", device)
    test = Split("test", device)
    print(f"device={device} train={len(train)} val={len(val)} test={len(test)}")

    frontend = LogMel().to(device)
    frontend.fit_norm(x for x, _ in val.batches(args.batch_size))
    model = DSCNN().to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    CKPT_DIR.mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y in train.batches(args.batch_size):
            loss = loss_fn(model(frontend(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        model.eval()
        acc, _ = evaluate(lambda x: model(frontend(x)), val, args.batch_size)
        print(f"epoch {epoch + 1:3d}  loss {total_loss / n:.4f}  val {acc:.4f}  {time.time() - t0:.0f}s")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "frontend": frontend.state_dict()},
                       CKPT_DIR / args.out)

    ckpt = torch.load(CKPT_DIR / args.out)
    model.load_state_dict(ckpt["model"])
    model.eval()
    acc, confusion = evaluate(lambda x: model(frontend(x)), test, args.batch_size)
    print(f"\nbest val {best:.4f}  test {acc:.4f}")
    print_confusion(confusion)


if __name__ == "__main__":
    main()
