"""Quantization-aware DS-CNN and its integer-only reference.

Number format (everything the RTL has to implement):
  - Input features: int8, value = q * 2^-f_in. Hardware: (log2mel - mean) << f_in, round, clamp.
  - Conv weights: int8 with a power-of-two scale per output channel, value = q * 2^-fw[c].
    BatchNorm and the input 1/std are folded in, so there is no separate normalization.
  - Conv bias: int32 at the accumulator scale 2^-(f_in + fw[c]).
  - Activations after ReLU: uint8, value = q * 2^-f_out.
  - Requantization: y = clamp((acc + 2^(s-1)) >> s, 0, 255) with s = f_in + fw[c] - f_out,
    i.e. a per-channel right shift with round-half-up. No multipliers.
  - Global average pool is a plain sum (the /500 folds into the FC bias).
  - FC: int8 weights with one scale for all classes, so the integer logits are
    comparable and argmax needs no rescaling.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .features import LogMel
from .model import DSCNN

POOL_SIZE = 25 * 20  # spatial size entering global average pool


def fake_quant(x, scale, lo, hi):
    """Round-half-up quantization at `scale` (x * scale -> integer) with a straight-through gradient."""
    q = torch.clamp(torch.floor(x * scale + 0.5), lo, hi) / scale
    return x + (q - x).detach()


def weight_frac(w):
    """Largest power-of-two fraction bits per output channel so int8 does not clip."""
    m = w.detach().abs().flatten(1).amax(1).clamp(min=1e-8)
    return torch.floor(torch.log2(127.0 / m))


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * s.view(-1, 1, 1, 1), bn.bias - bn.running_mean * s


class QConv(nn.Module):
    """Conv with BN folded in, followed by ReLU."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        w, b = fold_bn(conv, bn)
        self.weight = nn.Parameter(w.detach().clone())
        self.bias = nn.Parameter(b.detach().clone())
        self.stride, self.padding, self.groups = conv.stride, conv.padding, conv.groups
        self.register_buffer("f_out", torch.tensor(0.0))

    def forward(self, x, f_in, quant):
        w, b = self.weight, self.bias
        if quant:
            fw = weight_frac(w)
            w = fake_quant(w, 2.0 ** fw.view(-1, 1, 1, 1), -128, 127)
            b = fake_quant(b, 2.0 ** (f_in + fw), -(2**31), 2**31 - 1)
        y = F.relu(F.conv2d(x, w, b, self.stride, self.padding, groups=self.groups))
        if quant:
            y = fake_quant(y, 2.0 ** self.f_out, 0, 255)
        return y


class QDSCNN(nn.Module):
    """Takes raw audio, returns float logits. `quant=False` runs the BN-folded float model."""

    def __init__(self, model: DSCNN, frontend: LogMel):
        super().__init__()
        self.frontend = frontend.requires_grad_(False)
        convs = [model.stem]
        for block in model.blocks:
            convs.extend(block)
        self.layers = nn.ModuleList(QConv(seq[0], seq[1]) for seq in convs)
        # Fold input normalization (x - mean) / std into the stem. Padding stays exact
        # because zero in the (x - mean) domain is still zero.
        self.layers[0].weight.data /= frontend.std
        self.fc_weight = nn.Parameter(model.fc.weight.detach().clone())
        self.fc_bias = nn.Parameter(model.fc.bias.detach().clone())
        self.register_buffer("f_in", torch.tensor(0.0))
        self.quant = False

    def features(self, audio):
        """Mean-subtracted log2 mel, (B, 1, 49, 40)."""
        return (self.frontend.raw(audio) - self.frontend.mean).unsqueeze(1)

    def forward(self, audio, return_acts=False):
        x = self.features(audio)
        if self.quant:
            x = fake_quant(x, 2.0 ** self.f_in, -128, 127)
        acts = [x]
        f = self.f_in
        for layer in self.layers:
            x = layer(x, f, self.quant)
            f = layer.f_out
            acts.append(x)
        pooled = x.mean((2, 3))
        w, b = self.fc_weight, self.fc_bias
        if self.quant:
            fw = weight_frac(w).min()  # one scale for all classes
            w = fake_quant(w, 2.0 ** fw, -128, 127)
            b = fake_quant(b, POOL_SIZE * 2.0 ** (f + fw), -(2**31), 2**31 - 1)
        logits = F.linear(pooled, w, b)
        return (logits, acts) if return_acts else logits

    @torch.no_grad()
    def calibrate(self, batches, candidates=range(-4, 13)):
        """Pick power-of-two fraction bits for the input and each activation by minimum MSE."""
        self.quant = False
        collected = None
        for audio in batches:
            _, acts = self(audio, return_acts=True)
            sample = [a.flatten()[torch.randperm(a.numel(), device=a.device)[:200_000]] for a in acts]
            collected = sample if collected is None else [torch.cat(p) for p in zip(collected, sample)]

        def best_frac(x, lo, hi):
            errs = [((fake_quant(x, 2.0**f, lo, hi) - x) ** 2).mean().item() for f in candidates]
            return float(list(candidates)[min(range(len(errs)), key=errs.__getitem__)])

        self.f_in.fill_(best_frac(collected[0], -128, 127))
        for layer, x in zip(self.layers, collected[1:]):
            layer.f_out.fill_(best_frac(x, 0, 255))
        self.quant = True

    @torch.no_grad()
    def to_int(self):
        """Export integer parameters for the reference model / RTL."""
        f = self.f_in
        layers = []
        for layer in self.layers:
            fw = weight_frac(layer.weight)
            layers.append({
                "w": torch.floor(layer.weight * 2.0 ** fw.view(-1, 1, 1, 1) + 0.5).clamp(-128, 127).to(torch.int8).cpu(),
                "b": torch.floor(layer.bias * 2.0 ** (f + fw) + 0.5).clamp(-(2**31), 2**31 - 1).to(torch.int32).cpu(),
                "shift": (f + fw - layer.f_out).to(torch.int32).cpu(),
                "stride": layer.stride, "padding": layer.padding, "groups": layer.groups,
            })
            f = layer.f_out
        fw = weight_frac(self.fc_weight).min()
        return {
            "f_in": int(self.f_in),
            "mean": float(self.frontend.mean),
            "layers": layers,
            "fc_w": torch.floor(self.fc_weight * 2.0**fw + 0.5).clamp(-128, 127).to(torch.int8).cpu(),
            "fc_b": torch.floor(self.fc_bias * POOL_SIZE * 2.0 ** (f + fw) + 0.5).to(torch.int32).cpu(),
        }


def quantize_input(features, f_in):
    """Mean-subtracted log2 mel (float) -> int8 feature map, as the frontend RTL will produce."""
    return torch.clamp(torch.floor(features * 2.0**f_in + 0.5), -128, 127).to(torch.int64)


def int_forward(params, x, return_acts=False):
    """Integer-only inference. x: int64 (B, 1, 49, 40) in int8 range. Returns int64 logits,
    plus (per-layer outputs, pooled sums) if `return_acts`.

    Convs run in float64 on CPU, which is exact here (|acc| < 2^53)."""
    x = x.cpu()
    acts = []
    for p in params["layers"]:
        acc = F.conv2d(x.double(), p["w"].double(), p["b"].double(),
                       p["stride"], p["padding"], groups=p["groups"])
        acc = acc.round().to(torch.int64)
        s = p["shift"].to(torch.int64).view(1, -1, 1, 1)
        assert (s >= 1).all(), "left shifts not expected"
        x = ((acc + (torch.ones_like(s) << (s - 1))) >> s).clamp(0, 255)  # ReLU folded into the lower clamp
        acts.append(x)
    pooled = x.sum((2, 3))
    logits = pooled @ params["fc_w"].to(torch.int64).T + params["fc_b"].to(torch.int64)
    return (logits, acts, pooled) if return_acts else logits


def describe(params):
    lines = [f"input: int8, f_in={params['f_in']}"]
    for i, p in enumerate(params["layers"]):
        s = p["shift"]
        lines.append(f"layer {i}: w{tuple(p['w'].shape)} groups={p['groups']} "
                     f"shift {int(s.min())}..{int(s.max())}  |b|max={int(p['b'].abs().max())}")
    lines.append(f"fc: w{tuple(params['fc_w'].shape)}  |b|max={int(params['fc_b'].abs().max())}")
    return "\n".join(lines)

