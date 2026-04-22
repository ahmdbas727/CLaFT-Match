# CLaFTMatch_train.py
# Lightweight CLaFTMatch (FixMatch + R-ADC + CL-CAT) WITHOUT RA-TM
# UPDATED (requested):
#   (1) FixMatch-style training budget: fixed "iters_per_epoch" (default 500)
#       - decouples training from len(labeled_loader) so it matches common SSL protocols.
#   (2) CBCW: Class-Balanced Consistency Weighting (lightweight, ~10 lines)
#       - boosts unsupervised loss for tail/starved pseudo-classes based on acceptance deficit.
#   (3) REAL FIX kept: confidence-weighted EMA for p_u_hat
#   (4) Logs imbalance metrics + pred_dist (diagnostic)
#   (5) Windows-safe (pickle-safe RandAugment, no lambdas)
#   (6) NEW: EMA teacher for pseudo-labels (stabilizes late-stage drift; minimal lightweight fix)

import os
import math
import time
import json
import random
import argparse
import copy
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def sigmoid_rampup(current: int, rampup_length: int) -> float:
    if rampup_length == 0:
        return 1.0
    current = max(0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, ema_decay: float):
    """EMA update for teacher model.
    - EMA for floating tensors
    - hard copy for non-floating tensors (e.g., BN num_batches_tracked)
    """
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        src = msd[k]
        if torch.is_floating_point(v):
            v.mul_(ema_decay).add_(src, alpha=1.0 - ema_decay)
        else:
            v.copy_(src)
    ema_model.load_state_dict(esd, strict=False)

# ---------------------------
# WideResNet
# ---------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (None if self.equalInOut else
                             nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                       padding=0, bias=False))

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        shortcut = x if self.equalInOut else self.convShortcut(x)
        return shortcut + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10, dropRate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, dropRate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, dropRate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nStages = nStages[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).view(-1, self.nStages)
        return self.fc(out)

# ---------------------------
# Pickle-safe RandAugment
# ---------------------------

def ra_autocontrast(img: Image.Image, v): return TF.autocontrast(img)
def ra_equalize(img: Image.Image, v):     return TF.equalize(img)
def ra_rotate(img: Image.Image, v):       return TF.rotate(img, v)
def ra_solarize(img: Image.Image, v):     return TF.solarize(img, v)
def ra_color(img: Image.Image, v):        return TF.adjust_saturation(img, 1.0 + v)
def ra_contrast(img: Image.Image, v):     return TF.adjust_contrast(img, 1.0 + v)
def ra_brightness(img: Image.Image, v):   return TF.adjust_brightness(img, 1.0 + v)
def ra_sharpness(img: Image.Image, v):    return TF.adjust_sharpness(img, 1.0 + v)
def ra_shear_x(img: Image.Image, v):      return TF.affine(img, angle=0.0, translate=[0,0], scale=1.0, shear=[v, 0.0])
def ra_shear_y(img: Image.Image, v):      return TF.affine(img, angle=0.0, translate=[0,0], scale=1.0, shear=[0.0, v])
def ra_translate_x(img: Image.Image, v):  return TF.affine(img, angle=0.0, translate=[int(v),0], scale=1.0, shear=[0.0,0.0])
def ra_translate_y(img: Image.Image, v):  return TF.affine(img, angle=0.0, translate=[0,int(v)], scale=1.0, shear=[0.0,0.0])

class RandAugmentMC:
    def __init__(self, n: int = 2, m: int = 10):
        self.n = n
        self.m = m
        self.ops = [
            ("autocontrast", ra_autocontrast),
            ("equalize", ra_equalize),
            ("rotate", ra_rotate),
            ("solarize", ra_solarize),
            ("color", ra_color),
            ("contrast", ra_contrast),
            ("brightness", ra_brightness),
            ("sharpness", ra_sharpness),
            ("shear_x", ra_shear_x),
            ("shear_y", ra_shear_y),
            ("translate_x", ra_translate_x),
            ("translate_y", ra_translate_y),
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = random.sample(self.ops, self.n)
        rot_deg = int(self.m * 3)
        sol_thr = int(256 - self.m * 10)
        col_amt = self.m / 30.0
        shr_deg = self.m / 3.0
        trn_px  = int(self.m)

        for name, fn in ops:
            if name == "rotate":
                v = random.choice([-rot_deg, rot_deg])
            elif name == "solarize":
                v = max(0, min(255, sol_thr))
            elif name in ["color", "contrast", "brightness", "sharpness"]:
                v = random.uniform(0.0, col_amt)
            elif name in ["shear_x", "shear_y"]:
                v = random.choice([-shr_deg, shr_deg])
            elif name in ["translate_x", "translate_y"]:
                v = random.choice([-trn_px, trn_px])
            else:
                v = 0
            img = fn(img, v)
        return img

# ---------------------------
# Augmentations / Datasets
# ---------------------------

def get_transforms_cifar():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2471, 0.2435, 0.2616)

    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        RandAugmentMC(n=2, m=10),
    ])
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return weak, strong, to_tensor

class SSLUnlabeledCIFAR(Dataset):
    def __init__(self, base_dataset, indices, weak_aug, strong_aug, to_tensor):
        self.base = base_dataset
        self.indices = indices
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.base[real_idx]
        xw = self.to_tensor(self.weak_aug(img))
        xs = self.to_tensor(self.strong_aug(img))
        return xw, xs

class SSLLabeledCIFAR(Dataset):
    def __init__(self, base_dataset, indices, to_tensor):
        self.base = base_dataset
        self.indices = indices
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, y = self.base[real_idx]
        x = self.to_tensor(img)
        return x, y

# ---------------------------
# Imbalanced labeled split
# ---------------------------

def make_longtail_counts(num_classes: int, total: int, imb_ratio: float) -> List[int]:
    if imb_ratio <= 1.0:
        base = total // num_classes
        counts = [base] * num_classes
        counts[0] += total - sum(counts)
        return counts

    denom = sum([imb_ratio ** (-i/(num_classes-1)) for i in range(num_classes)])
    max_count = total / denom
    counts = [int(round(max_count * (imb_ratio ** (-i/(num_classes-1))))) for i in range(num_classes)]

    diff = total - sum(counts)
    i = 0
    while diff != 0:
        j = i % num_classes
        if diff > 0:
            counts[j] += 1
            diff -= 1
        else:
            if counts[j] > 1:
                counts[j] -= 1
                diff += 1
        i += 1
    return counts

def split_labeled_unlabeled(dataset, num_classes: int, num_labels: int, imb_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    set_seed(seed)
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    labeled_idx, unlabeled_idx = [], []

    class_indices = [torch.where(targets == c)[0].tolist() for c in range(num_classes)]
    for c in range(num_classes):
        random.shuffle(class_indices[c])

    per_class_counts = make_longtail_counts(num_classes, num_labels, imb_ratio)
    for c in range(num_classes):
        take = min(per_class_counts[c], len(class_indices[c]))
        labeled_idx.extend(class_indices[c][:take])
        unlabeled_idx.extend(class_indices[c][take:])

    random.shuffle(labeled_idx)
    random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx

def compute_class_weights(labeled_targets: List[int], num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for y in labeled_targets:
        counts[int(y)] += 1.0
    w = 1.0 / (counts + eps)
    w = w / (w.mean() + eps)
    return w

# ---------------------------
# Metrics
# ---------------------------

@torch.no_grad()
def compute_confusion_matrix(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[int(t), int(p)] += 1
    return cm

def metrics_from_cm(cm: torch.Tensor, eps: float = 1e-12) -> Dict[str, object]:
    cmf = cm.to(torch.float64)
    tp = torch.diag(cmf)
    support = cmf.sum(dim=1)
    pred_count = cmf.sum(dim=0)

    recall = tp / (support + eps)
    precision = tp / (pred_count + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    acc = tp.sum() / (cmf.sum() + eps)
    bal_acc = recall.mean()
    macro_f1 = f1.mean()

    pred_dist = (pred_count / (pred_count.sum() + eps)).cpu().tolist()

    return {
        "acc": float(acc.item()),
        "balanced_acc": float(bal_acc.item()),
        "macro_f1": float(macro_f1.item()),
        "per_class_recall": recall.cpu().tolist(),
        "per_class_precision": precision.cpu().tolist(),
        "per_class_f1": f1.cpu().tolist(),
        "support": support.cpu().tolist(),
        "pred_dist": pred_dist,
    }

# ---------------------------
# Config
# ---------------------------

@dataclass
class BMConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    out_dir: str = "./runs/CLaFTMatch"
    seed: int = 42

    num_labels: int = 4000
    imb_ratio: float = 100.0

    mu: int = 7
    batch_size: int = 64
    num_workers: int = 1

    wrn_depth: int = 28
    wrn_width: int = 2
    dropout: float = 0.0

    # FixMatch-like step budget
    epochs: int = 500
    iters_per_epoch: int = 500

    lr: float = 0.03
    weight_decay: float = 5e-4
    momentum: float = 0.9

    lambda_u_max: float = 1.0
    rampup_epochs: int = 5

    # EMA Teacher (for pseudo-labels; stabilizes late-stage drift)
    use_ema_teacher: int = 1
    ema_decay: float = 0.999

    # R-ADC
    ema_m: float = 0.999
    gamma_calib: float = 1.5
    eps: float = 1e-6
    prior: str = "uniform"  # "uniform" or "labeled"
    p_ema_alpha: float = 1.5  # confidence exponent (lower -> more inclusive)

    # CL-CAT
    tau_init: float = 0.88
    tau_min: float = 0.30
    tau_max: float = 0.99
    beta_target: float = 0.0
    eta_ctrl: float = 0.03
    rho_accept_ema: float = 0.90

    # Stability
    tau_inertia: float = 0.70
    tau_step_limit: float = 0.05
    warmup_enable_ctrl_epochs: int = 1

    # CBCW (Class-Balanced Consistency Weighting)
    cbcw_enable: int = 1
    cbcw_lambda: float = 1.0   # strength of weighting
    cbcw_clip: float = 3.0     # avoid extreme weights
    cbcw_warmup_epochs: int = 1

    # Model selection
    save_best_by: str = "acc"  # "balanced_acc" or "acc" or "macro_f1"

def get_num_classes(dataset_name: str) -> int:
    return 10 if dataset_name.lower() == "cifar10" else 100

def build_pi(cfg: BMConfig, labeled_targets: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    if cfg.prior == "uniform":
        return torch.ones(num_classes, device=device) / num_classes
    if cfg.prior == "labeled":
        counts = torch.zeros(num_classes, device=device)
        for y in labeled_targets:
            counts[int(y)] += 1.0
        return counts / (counts.sum() + cfg.eps)
    raise ValueError("prior must be 'uniform' or 'labeled'")

# ---------------------------
# Training
# ---------------------------

def train(cfg: BMConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_dir(cfg.out_dir)
    save_json(os.path.join(cfg.out_dir, "config.json"), cfg.__dict__)

    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    num_classes = get_num_classes(cfg.dataset)

    # Load datasets
    if cfg.dataset.lower() == "cifar10":
        base_train = datasets.CIFAR10(cfg.data_root, train=True, download=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        testset = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=test_transform)
    else:
        base_train = datasets.CIFAR100(cfg.data_root, train=True, download=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        testset = datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=test_transform)

    weak_aug, strong_aug, to_tensor = get_transforms_cifar()
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(
        base_train, num_classes, cfg.num_labels, cfg.imb_ratio, cfg.seed
    )
    labeled_targets = [base_train.targets[i] for i in labeled_idx]

    labeled_set = SSLLabeledCIFAR(base_train, labeled_idx, to_tensor)
    unlabeled_set = SSLUnlabeledCIFAR(base_train, unlabeled_idx, weak_aug, strong_aug, to_tensor)

    pin = torch.cuda.is_available()
    labeled_loader = DataLoader(
        labeled_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(cfg.num_workers > 0)
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=cfg.batch_size * cfg.mu, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=(cfg.num_workers > 0)
    )
    test_loader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=max(0, min(cfg.num_workers, 2)), pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0)
    )

    model = WideResNet(depth=cfg.wrn_depth, widen_factor=cfg.wrn_width,
                       num_classes=num_classes, dropRate=cfg.dropout).to(device)

    # EMA teacher model for generating pseudo-labels (lightweight stability fix)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    w_cls = compute_class_weights(labeled_targets, num_classes).to(device)

    # R-ADC state
    p_u_hat = torch.ones(num_classes, device=device) / num_classes

    # CL-CAT state
    tau_c = torch.ones(num_classes, device=device) * cfg.tau_init
    r_c = torch.zeros(num_classes, device=device)

    pi = build_pi(cfg, labeled_targets, num_classes, device=device)
    t_c = (pi ** cfg.beta_target)
    t_c = t_c / (t_c.sum() + cfg.eps)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        lambda_u = cfg.lambda_u_max * sigmoid_rampup(epoch - 1, cfg.rampup_epochs)
        ctrl_enabled = (epoch > cfg.warmup_enable_ctrl_epochs)

        running = {"loss": 0.0, "loss_l": 0.0, "loss_u": 0.0, "mask_rate": 0.0, "acc_l": 0.0}
        steps = 0

        for it in range(cfg.iters_per_epoch):
            try:
                xl, yl = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                xl, yl = next(labeled_iter)

            try:
                xw, xs = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                xw, xs = next(unlabeled_iter)

            xl = xl.to(device, non_blocking=True)
            yl = yl.to(device, non_blocking=True)
            xw = xw.to(device, non_blocking=True)
            xs = xs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Supervised
                logits_l = model(xl)
                loss_l = F.cross_entropy(logits_l, yl, weight=w_cls)
                acc_l = float((logits_l.argmax(dim=1) == yl).float().mean().item())

                with torch.no_grad():
                    # Teacher generates pseudo-labels (key stability fix)
                    logits_w = ema_model(xw) if cfg.use_ema_teacher else model(xw)
                    q = F.softmax(logits_w, dim=1)
                    conf_q, _ = q.max(dim=1)

                    # confidence-weighted marginal EMA
                    w = conf_q.clamp(0.0, 1.0).pow(cfg.p_ema_alpha)
                    p_batch = (q * w.unsqueeze(1)).sum(dim=0) / (w.sum() + cfg.eps)
                    p_u_hat = cfg.ema_m * p_u_hat + (1.0 - cfg.ema_m) * p_batch
                    p_u_hat = p_u_hat / (p_u_hat.sum() + cfg.eps)

                    # calibration
                    g = ((pi / (p_u_hat + cfg.eps)) ** cfg.gamma_calib).clamp(min=cfg.eps, max=50.0)
                    q_tilde = q * g.unsqueeze(0)
                    q_tilde = q_tilde / (q_tilde.sum(dim=1, keepdim=True) + cfg.eps)

                    s, yhat = q_tilde.max(dim=1)
                    tau_eff = tau_c[yhat] if ctrl_enabled else torch.full_like(s, cfg.tau_init)
                    mask = (s >= tau_eff).float()
                    mask_rate = float(mask.mean().item())

                logits_s = model(xs)

                # CBCW
                per_sample_u = F.cross_entropy(logits_s, yhat, reduction="none")
                if cfg.cbcw_enable and ctrl_enabled and epoch > cfg.cbcw_warmup_epochs:
                    deficit = (t_c - r_c).clamp(min=0.0)
                    w_c = (1.0 + cfg.cbcw_lambda * deficit / (t_c + cfg.eps)).clamp(1.0, cfg.cbcw_clip)
                    w_samp = w_c[yhat]
                    loss_u = (per_sample_u * w_samp * mask).sum() / (mask.sum() + cfg.eps)
                else:
                    loss_u = (per_sample_u * mask).mean()

                loss = loss_l + lambda_u * loss_u

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if cfg.use_ema_teacher:
                update_ema(ema_model, model, cfg.ema_decay)

            # CL-CAT update
            if ctrl_enabled:
                with torch.no_grad():
                    denom = torch.zeros(num_classes, device=device)
                    num = torch.zeros(num_classes, device=device)
                    denom.scatter_add_(0, yhat, torch.ones_like(mask))
                    num.scatter_add_(0, yhat, mask)

                    batch_rate = num / (denom + cfg.eps)
                    r_c = cfg.rho_accept_ema * r_c + (1.0 - cfg.rho_accept_ema) * batch_rate

                    tau_new = tau_c * torch.exp(cfg.eta_ctrl * (r_c - t_c))
                    tau_new = tau_new.clamp(min=cfg.tau_min, max=cfg.tau_max)

                    tau_smooth = cfg.tau_inertia * tau_c + (1.0 - cfg.tau_inertia) * tau_new
                    if cfg.tau_step_limit > 0:
                        tau_smooth = torch.max(torch.min(tau_smooth, tau_c + cfg.tau_step_limit),
                                               tau_c - cfg.tau_step_limit)

                    tau_c = tau_smooth.clamp(min=cfg.tau_min, max=cfg.tau_max)

            running["loss"] += float(loss.item())
            running["loss_l"] += float(loss_l.item())
            running["loss_u"] += float(loss_u.item())
            running["mask_rate"] += mask_rate
            running["acc_l"] += acc_l
            steps += 1

        scheduler.step()
        for k in running:
            running[k] /= max(1, steps)

        # Eval (use EMA model when enabled)
        eval_net = ema_model if cfg.use_ema_teacher else model
        cm = compute_confusion_matrix(eval_net, test_loader, num_classes, device)
        m = metrics_from_cm(cm)

        if cfg.save_best_by == "balanced_acc":
            score = m["balanced_acc"]
        elif cfg.save_best_by == "macro_f1":
            score = m["macro_f1"]
        else:
            score = m["acc"]

        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict() if cfg.use_ema_teacher else None,
                "optimizer": optimizer.state_dict(),
                "p_u_hat": p_u_hat.detach().cpu(),
                "tau_c": tau_c.detach().cpu(),
                "r_c": r_c.detach().cpu(),
                "cfg": cfg.__dict__,
                "best_score": best_score,
                "best_by": cfg.save_best_by,
            }, os.path.join(cfg.out_dir, "best.pt"))

        log_console = {
            "epoch": epoch,
            "time_sec": round(time.time() - t0, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lambda_u": float(lambda_u),
            "train_loss": round(running["loss"], 4),
            "loss_l": round(running["loss_l"], 4),
            "loss_u": round(running["loss_u"], 4),
            "mask_rate": round(running["mask_rate"], 4),
            "acc_l": round(running["acc_l"], 4),
            "test_acc": round(m["acc"], 4),
            "balanced_acc": round(m["balanced_acc"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "best_score": round(best_score, 4),
            "best_by": cfg.save_best_by,
            "ctrl_enabled": bool(ctrl_enabled),
            "iters_per_epoch": cfg.iters_per_epoch,
            "cbcw_enable": int(cfg.cbcw_enable),
            "use_ema_teacher": int(cfg.use_ema_teacher),
            "ema_decay": float(cfg.ema_decay),
        }
        print(json.dumps(log_console))

        append_jsonl(metrics_path, {
            **log_console,
            "per_class_recall": m["per_class_recall"],
            "per_class_precision": m["per_class_precision"],
            "per_class_f1": m["per_class_f1"],
            "support": m["support"],
            "pred_dist": m["pred_dist"],
        })

    print(f"Done. Best {cfg.save_best_by}: {best_score:.4f}")
    print(f"Checkpoint: {os.path.join(cfg.out_dir, 'best.pt')}")
    print(f"Metrics log: {metrics_path}")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/CLaFTMatch_cbcw")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_labels", type=int, default=4000)
    p.add_argument("--imb_ratio", type=float, default=100.0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mu", type=int, default=7)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--iters_per_epoch", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--num_workers", type=int, default=1)

    # R-ADC
    p.add_argument("--ema_m", type=float, default=0.999)
    p.add_argument("--gamma_calib", type=float, default=1.5)
    p.add_argument("--prior", type=str, default="uniform", choices=["uniform", "labeled"])
    p.add_argument("--p_ema_alpha", type=float, default=1.5)

    # CL-CAT
    p.add_argument("--tau_init", type=float, default=0.88)
    p.add_argument("--tau_min", type=float, default=0.30)
    p.add_argument("--tau_max", type=float, default=0.99)
    p.add_argument("--beta_target", type=float, default=0.0)
    p.add_argument("--eta_ctrl", type=float, default=0.03)
    p.add_argument("--rho_accept_ema", type=float, default=0.90)

    # Stabilizers
    p.add_argument("--tau_inertia", type=float, default=0.70)
    p.add_argument("--tau_step_limit", type=float, default=0.05)
    p.add_argument("--warmup_enable_ctrl_epochs", type=int, default=1)

    # Schedules
    p.add_argument("--lambda_u_max", type=float, default=1.0)
    p.add_argument("--rampup_epochs", type=int, default=5)

    # EMA Teacher (pseudo-label stability)
    p.add_argument("--use_ema_teacher", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # CBCW
    p.add_argument("--cbcw_enable", type=int, default=1)
    p.add_argument("--cbcw_lambda", type=float, default=1.0)
    p.add_argument("--cbcw_clip", type=float, default=3.0)
    p.add_argument("--cbcw_warmup_epochs", type=int, default=1)

    # Model selection
    p.add_argument("--save_best_by", type=str, default="macro_f1",
                   choices=["balanced_acc", "acc", "macro_f1"])

    return p.parse_args()

def main():
    a = parse_args()
    cfg = BMConfig(
        dataset=a.dataset,
        data_root=a.data_root,
        out_dir=a.out_dir,
        seed=a.seed,

        num_labels=a.num_labels,
        imb_ratio=a.imb_ratio,

        batch_size=a.batch_size,
        mu=a.mu,
        epochs=a.epochs,
        iters_per_epoch=a.iters_per_epoch,
        lr=a.lr,
        num_workers=a.num_workers,

        ema_m=a.ema_m,
        gamma_calib=a.gamma_calib,
        prior=a.prior,
        p_ema_alpha=a.p_ema_alpha,

        tau_init=a.tau_init,
        tau_min=a.tau_min,
        tau_max=a.tau_max,
        beta_target=a.beta_target,
        eta_ctrl=a.eta_ctrl,
        rho_accept_ema=a.rho_accept_ema,

        tau_inertia=a.tau_inertia,
        tau_step_limit=a.tau_step_limit,
        warmup_enable_ctrl_epochs=a.warmup_enable_ctrl_epochs,

        lambda_u_max=a.lambda_u_max,
        rampup_epochs=a.rampup_epochs,

        use_ema_teacher=a.use_ema_teacher,
        ema_decay=a.ema_decay,

        cbcw_enable=a.cbcw_enable,
        cbcw_lambda=a.cbcw_lambda,
        cbcw_clip=a.cbcw_clip,
        cbcw_warmup_epochs=a.cbcw_warmup_epochs,

        save_best_by=a.save_best_by
    )
    train(cfg)

if __name__ == "__main__":
    main()


# Y=50
# mu=7
# python CLaFTMatch_train_add_ema.py --dataset cifar10 --data_root ./data --out_dir ./runs/bm_c10_N1_1500_M1_3000_g50_80base_ema --num_labels 4200 --imb_ratio 50 --save_best_by acc --num_workers 1 --iters_per_epoch 512 --epochs 512 --tau_init 0.88 --tau_min 0.30 --eta_ctrl 0.03 --tau_inertia 0.70 --tau_step_limit 0.05 --gamma_calib 1.5 --p_ema_alpha 1.5 --cbcw_enable 1 --cbcw_lambda 1.0 --cbcw_clip 3.0 --use_ema_teacher 1 --ema_decay 0.9995

# mu=2
# python CLaFTMatch_train_add_ema.py --dataset cifar10 --data_root ./data --out_dir ./runs/bm_c10_N1_1500_M1_3000_g50_mu2_ema --num_labels 4200 --imb_ratio 50 --save_best_by acc --num_workers 1 --batch_size 64 --mu 2 --iters_per_epoch 512 --epochs 512 --lambda_u_max 1.0 --rampup_epochs 5 --tau_init 0.88 --tau_min 0.30 --eta_ctrl 0.03 --tau_inertia 0.70 --tau_step_limit 0.05 --gamma_calib 1.5 --p_ema_alpha 1.5 --cbcw_enable 1 --cbcw_lambda 1.0 --cbcw_clip 3.0 --use_ema_teacher 1 --ema_decay 0.9995

# Y=100
# mu=2
# python CLaFTMatch_train_add_ema.py --dataset cifar10 --data_root ./data --out_dir ./runs/bm_c10_N1_1500_M1_3000_g100_ema_stable1 --num_labels 3723 --imb_ratio 100 --save_best_by acc --num_workers 1 --batch_size 64 --mu 2 --iters_per_epoch 512 --epochs 512 --lambda_u_max 0.75 --rampup_epochs 20 --tau_init 0.88 --tau_min 0.30 --eta_ctrl 0.03 --tau_inertia 0.70 --tau_step_limit 0.05 --gamma_calib 1.5 --p_ema_alpha 1.5 --cbcw_enable 1 --cbcw_lambda 1.0 --cbcw_clip 3.0 --use_ema_teacher 1 --ema_decay 0.9995



# Y=150
# mu=2
# python CLaFTMatch_train_add_ema.py --dataset cifar10 --data_root ./data --out_dir ./runs/bm_c10_N1_1500_M1_3000_g150_ema_stable1 --num_labels 3500 --imb_ratio 150 --save_best_by acc --num_workers 1 --batch_size 64 --mu 2 --iters_per_epoch 512 --epochs 512 --lambda_u_max 0.65 --rampup_epochs 30 --tau_init 0.88 --tau_min 0.30 --eta_ctrl 0.03 --tau_inertia 0.70 --tau_step_limit 0.05 --gamma_calib 1.5 --p_ema_alpha 1.5 --cbcw_enable 1 --cbcw_lambda 1.2 --cbcw_clip 3.0 --use_ema_teacher 1 --ema_decay 0.9995