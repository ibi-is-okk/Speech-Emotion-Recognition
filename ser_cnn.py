"""
ser_cnn.py - the CNN model, SpecAugment and checkpoint helpers.

Used by: the Colab training / evaluation notebooks and the Streamlit website, so the model
that is trained is exactly the model that is loaded for predictions.
Requires torch. Audio preprocessing lives in ser_audio.py.
"""
import random
import torch
import torch.nn as nn

from ser_audio import EMOTIONS, PREPROCESSING


class SERCNN(nn.Module):
    """Small CNN for 1 x 128 x 173 log-mel spectrograms.

    Each block:  Conv3x3 -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout2d
    Head:        [global average pool ++ global max pool] -> Dropout -> Linear(n_classes)

    channels : output channels of each conv block (3 or 4 blocks are sensible)
    dropout  : final dropout; the dropout inside the blocks is half of it
    """

    def __init__(self, n_classes=8, channels=(32, 64, 128, 128), dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout * 0.5),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in, n_classes))

    def forward(self, x):                       # x: (batch, 1, 128, 173)
        x = self.features(x)                    # (batch, C, h, w)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)                # raw scores (logits), one per emotion


def spec_augment(x, n_freq=2, max_freq=20, n_time=2, max_time=25):
    """SpecAugment on a batch of normalized spectrograms, shape (B, 1, F, T).

    Blanks out n_freq random frequency bands (up to max_freq mel bins wide) and n_time random
    time spans (up to max_time frames wide) with 0, which equals the training mean after
    normalization. Use ONLY on training batches. Uses python's `random`, so it follows the seed.
    """
    x = x.clone()
    B, _, F, T = x.shape
    for b in range(B):
        for _ in range(n_freq):
            w = random.randint(0, max_freq)
            f0 = random.randint(0, F - w)
            x[b, :, f0:f0 + w, :] = 0.0
        for _ in range(n_time):
            w = random.randint(0, max_time)
            t0 = random.randint(0, T - w)
            x[b, :, :, t0:t0 + w] = 0.0
    return x


def save_checkpoint(path, model, cfg, norm_mean, norm_std, extra=None):
    """Save everything needed to use the model later, in ONE file."""
    cfg = dict(cfg)
    cfg["channels"] = list(cfg["channels"])
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg,
        "norm_mean": float(norm_mean),
        "norm_std": float(norm_std),
        "emotions": list(EMOTIONS),
        "preprocessing": dict(PREPROCESSING),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint file. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)   # our own trusted file
    cfg = ckpt["config"]
    model = SERCNN(n_classes=len(ckpt["emotions"]), channels=tuple(cfg["channels"]),
                   dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model, x, device="cpu"):
    """x: numpy or tensor of shape (n, 1, 128, 173), already normalized -> (n, 8) probabilities."""
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    return torch.softmax(model(x), dim=1).cpu().numpy()
