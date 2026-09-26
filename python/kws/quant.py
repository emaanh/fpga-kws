"""Quantization-aware DS-CNN and its integer-only reference.

Number format (everything the RTL has to implement):
  - Input features: int8, value = q * 2^-f_in. Hardware: (log2mel - mean) << f_in, round, clamp.
  - Conv weights: int8 with a power-of-two scale per output channel, value = q * 2^-fw[c].
    BatchNorm and the input 1/std are folded in, so there is no separate normalization.
  - Conv bias: int32 at the accumulator scale 2^-(f_in + fw[c]).
  - Activations after ReLU: uint8, value = q * 2^-f_out.
  - Requantization: y = clamp((acc + 2^(s-1)) >> s, 0, 255) with s = f_in + fw[c] - f_out,
    i.e. a per-channel right shift with round-half-up. No multipliers.
  - Global average pool is a plain sum (the 1/(rows*cols) folds into the FC bias).
  - FC: int8 weights with one scale for all classes, so the integer logits are
    comparable and argmax needs no rescaling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import N_FRAMES, N_MELS
from .features import LogMel
from .model import DSCNN


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
        with torch.no_grad():  # spatial size entering the global average pool
            dev = model.fc.weight.device
            self.pool_size = model.blocks(model.stem(torch.zeros(1, 1, N_FRAMES, N_MELS, device=dev))).shape[2:].numel()

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
            b = fake_quant(b, self.pool_size * 2.0 ** (f + fw), -(2**31), 2**31 - 1)
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
            "fc_b": torch.floor(self.fc_bias * self.pool_size * 2.0 ** (f + fw) + 0.5).to(torch.int32).cpu(),
        }


def quantize_input(features, f_in):
    """Mean-subtracted log2 mel (float) -> int8 feature map, as the frontend RTL will produce."""
    return torch.clamp(torch.floor(features * 2.0**f_in + 0.5), -128, 127).to(torch.int64)


def int_convs(params, x):
    """The integer conv layers on x (int64, (B, 1, H, 40)). Returns every layer's output.

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
    return acts


def int_forward(params, x, return_acts=False):
    """Integer-only inference. x: int64 (B, 1, 49, 40) in int8 range. Returns int64 logits,
    plus (per-layer outputs, pooled sums) if `return_acts`."""
    acts = int_convs(params, x)
    pooled = acts[-1].sum((2, 3))
    logits = pooled @ params["fc_w"].to(torch.int64).T + params["fc_b"].to(torch.int64)
    return (logits, acts, pooled) if return_acts else logits


def is_streaming(params):
    """True for models without padding along time (see model.py)."""
    return all(p["padding"][0] == 0 for p in params["layers"])


def pool_rows(params, n_frames=N_FRAMES):
    """Rows of the last layer for an n_frames window."""
    rows = n_frames
    for p in params["layers"]:
        kh, sh, ph = p["w"].shape[2], p["stride"][0], p["padding"][0]
        rows = (rows + 2 * ph - kh) // sh + 1
    return rows


def stream_forward(params, q):
    """Integer logits of a streaming model over a long feature stream q (int, (T, 40)).

    Result k is the window of frames 2k .. 2k+48, i.e. one result every 2 frames. With no
    padding along time, each layer's rows over the whole stream are the rows every window
    sees, so this equals int_forward on each window exactly. It is also what the streaming
    engine computes: one new row per layer every 2 frames, and the pool as a sum of the
    last pool_rows() rows of the last layer."""
    assert is_streaming(params)
    x = torch.as_tensor(q, dtype=torch.int64)[None, None]
    rowsum = int_convs(params, x)[-1][0].sum(2)            # (64, rows)
    pooled = rowsum.unfold(1, pool_rows(params), 1).sum(2)  # (64, results)
    return pooled.T @ params["fc_w"].to(torch.int64).T + params["fc_b"].to(torch.int64)


def logits_over_time(params, q, every=5):
    """Integer logits over a feature stream q (T, 40), as the board computes them.

    Returns (end frame of each window, logits). A streaming model gives a result every 2
    frames; an older full-window model is run on a window every `every` frames."""
    q = np.asarray(q)
    if is_streaming(params):
        logits = stream_forward(params, q)
        return 2 * np.arange(len(logits)) + N_FRAMES - 1, logits
    ends = np.arange(N_FRAMES, len(q) + 1, every)
    if len(ends) == 0:
        return ends - 1, torch.zeros(0, params["fc_w"].shape[0], dtype=torch.int64)
    x = torch.from_numpy(np.stack([q[k - N_FRAMES : k] for k in ends]).astype(np.int64)).unsqueeze(1)
    return ends - 1, torch.cat([int_forward(params, b) for b in x.split(256)])


def describe(params):
    lines = [f"input: int8, f_in={params['f_in']}"]
    for i, p in enumerate(params["layers"]):
        s = p["shift"]
        lines.append(f"layer {i}: w{tuple(p['w'].shape)} groups={p['groups']} "
                     f"shift {int(s.min())}..{int(s.max())}  |b|max={int(p['b'].abs().max())}")
    lines.append(f"fc: w{tuple(params['fc_w'].shape)}  |b|max={int(params['fc_b'].abs().max())}")
    return "\n".join(lines)

