"""Quantization-aware fine-tuning of the float DS-CNN, then integer-only evaluation.

    uv run python -m kws.qat              # needs checkpoints/dscnn_float.pt
    uv run python -m kws.qat --epochs 1   # smoke test
"""

import argparse
import itertools
import time

import torch
from torch import nn

from .config import CKPT_DIR
from .data import make_splits
from .features import LogMel
from .model import DSCNN
from .quant import QDSCNN, describe, int_forward, quantize_input
from .train import evaluate, pick_device, print_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gain-db", type=float, default=10.0, help="random speech gain range (+-dB)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--float-ckpt", default="dscnn_float.pt")
    p.add_argument("--out", default="dscnn_int8.pt")
    p.add_argument("--data", default="real", choices=["real", "tts", "tts+real"])
    p.add_argument("--extra-frac", type=float, default=0.2,
                   help="TTS recipes: unknown and silence clips per epoch, relative to keywords")
    p.add_argument("--realism", action="store_true")
    args = p.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    train, val, test = make_splits(args.data, device, gain_db=args.gain_db, realism=args.realism,
                                   extra_frac=args.extra_frac)

    ckpt = torch.load(CKPT_DIR / args.float_ckpt)
    model, frontend = DSCNN(), LogMel()
    model.load_state_dict(ckpt["model"])
    frontend.load_state_dict(ckpt["frontend"])
    model.eval()
    qm = QDSCNN(model, frontend).to(device)

    bs = args.batch_size
    acc, _ = evaluate(qm, val, bs)
    print(f"float, BN folded   val {acc:.4f}")
    qm.calibrate(x for x, _ in itertools.islice(train.batches(bs), 20))
    acc, _ = evaluate(qm, val, bs)
    print(f"post-training quant val {acc:.4f}  (f_in={int(qm.f_in)}, "
          f"f_out={[int(l.f_out) for l in qm.layers]})")

    opt = torch.optim.AdamW(qm.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for epoch in range(args.epochs):
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y in train.batches(bs):
            loss = loss_fn(qm(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        acc, _ = evaluate(qm, val, bs)
        print(f"qat epoch {epoch + 1:3d}  loss {total_loss / n:.4f}  val {acc:.4f}  {time.time() - t0:.0f}s")
        if acc > best:
            best = acc
            torch.save(qm.state_dict(), CKPT_DIR / args.out.replace("int8", "qat"))

    qm.load_state_dict(torch.load(CKPT_DIR / args.out.replace("int8", "qat")))
    params = qm.to_int()
    torch.save(params, CKPT_DIR / args.out)
    print("\n" + describe(params))

    def int_predict(x):
        return int_forward(params, quantize_input(qm.features(x), params["f_in"])).argmax(1)

    # How often does the integer model disagree with the fake-quant model it came from?
    mismatches = total = 0
    with torch.no_grad():
        for x, _ in test.batches(bs):
            mismatches += (qm(x).argmax(1).cpu() != int_predict(x)).sum().item()
            total += len(x)

    fq_acc, _ = evaluate(qm, test, bs)
    int_val, _ = evaluate(int_predict, val, bs)
    int_test, confusion = evaluate(int_predict, test, bs)
    print(f"\nfake-quant test {fq_acc:.4f}")
    print(f"integer    val  {int_val:.4f}  test {int_test:.4f}  "
          f"(disagrees with fake-quant on {mismatches}/{total} test clips)")
    print_confusion(confusion)


if __name__ == "__main__":
    main()
