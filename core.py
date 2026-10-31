"""SparseAttnNet: model, data, losses and metrics.

Model: a coarse convolutional network scores every pixel, the k highest-scoring pixels (k adapted during
training) are embedded as points and summarized by a linear-complexity fine attention, and the result is fused
with pooled coarse features for classification. Baselines: torchvision ResNet50 / MobileNetV2 and timm
ConvNeXt-Tiny, DeiT-Small, ViT-Small, Swin-Tiny. No data augmentation (resize only).
"""
import os, random, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SPARSEATTNNET_DATA', os.path.join(ROOT, 'data'))
CACHE = os.path.join(ROOT, 'cache')
TRANSFORMERS = ['ConvNeXt', 'DeiT', 'ViTSmall', 'SwinTiny']
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Class folders under data/, in label order. Files inside a folder are read in sorted (acquisition) order.
CLASS_DIRS = {
    'opd': ['Tcells/opd_mat/ACT', 'Tcells/opd_mat/ACT7d', 'Tcells/opd_mat/Tcells'],
    'wbc': ['WBC_Train/Basophil', 'WBC_Train/Eosinophil', 'WBC_Train/Lymphocyte',
            'WBC_Train/Monocyte', 'WBC_Train/Neutrophil'],
    'holos': ['Tcells/holos/ACT', 'Tcells/holos/ACT7d', 'Tcells/holos/Tcells'],
    'event': ['Tcells/event_new/SW480', 'Tcells/event_new/SW620'],
}


class Config:
    SEED = 123456
    MAX_EPOCHS = 200          # fixed budget: every model trains exactly MAX_EPOCHS, no early stopping
    INNER_VAL = 0.2
    INNER_SPLIT = 'contiguous'  # last per-class 20 % block of the outer-train set ('random' for WBC, see run.py)
    MASK_LEARNING_RATE = 1e-3
    LR_K = 10
    K_MOMENTUM = 0.2
    EMA_BETA = 0.2
    DISTILLATION_WARMUP_EPOCHS = 5
    DISTILLATION_WEIGHT = 0.1
    EMPHASIS_FACTOR = 2.0
    CONTRASTIVE_WEIGHT = 0.1
    CONTRASTIVE_TEMPERATURE = 0.07
    FOCAL_LOSS_GAMMA = 2.0
    BORDER = 5
    # ablation switches (set from the command line in run.py)
    FIXED_K = None
    FULL_IMAGE = False
    RANDOM_SELECT = False
    COARSE_ONLY = False
    FINE_ONLY = False
    PRETRAINED = False

    @staticmethod
    def update(dataset_type, model_type):
        Config.DATASET_TYPE = dataset_type
        Config.MODEL_TYPE = model_type
        if model_type in TRANSFORMERS:
            Config.IMAGE_SIZE = 224
        elif dataset_type == 'wbc':
            Config.IMAGE_SIZE = 256
        else:
            Config.IMAGE_SIZE = 135
        Config.LEARNING_RATE = 5e-5 if model_type in TRANSFORMERS else 1e-3
        Config.NUM_CLASSES = 5 if dataset_type == 'wbc' else 2 if dataset_type == 'event' else 3
        Config.INITIAL_K = 8000 if dataset_type == 'wbc' else 3000
        Config.K_MIN = 1500
        Config.K_MAX = 135 * 135
        Config.HIDDEN_DIM = 512
        Config.NUM_HEADS = 4
        Config.EMBED_DIM = 16
        Config.MASK_FEATURE_DIM = 16
        Config.BATCH_SIZE = 32 if dataset_type == 'wbc' else 64
        Config.FOCAL_LOSS_ALPHA = ([1.0] * 5 if dataset_type == 'wbc' else
                                   [1.0, 1.0] if dataset_type == 'event' else [2.0, 2.0, 1.0])


def set_seed(seed=None):
    seed = Config.SEED if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def list_samples(dataset_type):
    samples = []
    for ci, d in enumerate(CLASS_DIRS[dataset_type]):
        p = os.path.join(DATA, d)
        for f in sorted(os.listdir(p)):
            samples.append((os.path.join(p, f), ci))
    return samples


def _load_uint8(path, dataset_type, size):
    """Read one image and resize it (Lanczos) to size x size, as uint8. OPD maps (.mat) are min-max scaled."""
    if dataset_type == 'opd':
        a = loadmat(path)['opd_crop'].astype(np.float32)
        mn, mx = a.min(), a.max()
        a = (a - mn) / (mx - mn) * 255.0 if mx > mn else a - mn
        img = Image.fromarray(a.astype(np.uint8), mode='L')
    elif dataset_type == 'wbc':
        img = Image.open(path).convert('RGB')
    else:
        img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    a = np.array(img)
    return a.transpose(2, 0, 1) if a.ndim == 3 else a


def load_cache(dataset_type, size):
    """All images of a dataset at one resolution, cached in cache/<dataset>_<size>.npz after the first call."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f'{dataset_type}_{size}.npz')
    samples = list_samples(dataset_type)
    paths = [os.path.relpath(p, DATA) for p, _ in samples]
    labels = np.array([l for _, l in samples], dtype=np.int64)
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=False)
        assert list(z['paths']) == paths, 'cache file list mismatch (delete the cache file and rerun)'
        return z['x'], labels, paths
    x = np.stack([_load_uint8(p, dataset_type, size) for p, _ in samples])
    np.savez(fn, x=x, paths=np.array(paths))
    return x, labels, paths


class CachedDataset(Dataset):
    def __init__(self, x_uint8, labels, idx, dataset_type, model_type):
        self.x, self.y, self.idx = x_uint8, labels, np.asarray(idx)
        self.ds, self.mt = dataset_type, model_type

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        t = torch.from_numpy(self.x[j].astype(np.float32) / 255.0)
        if self.mt != 'ours':
            if t.dim() == 2:
                t = t.unsqueeze(0).repeat(3, 1, 1)  # grayscale replicated to 3 channels for the baselines
            if Config.PRETRAINED:  # ImageNet normalization for ImageNet-pretrained baselines only
                t = (t - IMAGENET_MEAN) / IMAGENET_STD
        elif self.ds == 'opd':
            t = t.unsqueeze(0)
        return t, torch.tensor(self.y[j]).long()


def outer_folds(labels, shuffle=False):
    """Flow-imaging sets: StratifiedKFold(5) without shuffling, i.e. contiguous blocks in acquisition order, so
    consecutive projections of the same cell stay in one fold. WBC (independent single-cell images): shuffled
    stratified 5-fold with seed 123456."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED) if shuffle else StratifiedKFold(n_splits=5, shuffle=False)
    return [(tr, te) for tr, te in skf.split(np.arange(len(labels)), labels)]


def inner_split(train_idx, labels):
    """Inner validation split (20 %) of the outer-training indices. 'contiguous' mirrors the outer protocol: the
    last 20 % of each class in acquisition order. 'random' (WBC): stratified random split, seed 123456."""
    if Config.INNER_SPLIT == 'random':
        tr, va = train_test_split(train_idx, test_size=Config.INNER_VAL, stratify=labels[train_idx],
                                  random_state=Config.SEED)
        return np.sort(tr), np.sort(va)
    train_idx = np.sort(np.asarray(train_idx))
    blocks = list(StratifiedKFold(n_splits=5, shuffle=False).split(train_idx, labels[train_idx]))
    tr, va = blocks[-1]
    return train_idx[tr], train_idx[va]


def contiguous_runs(idx, labels, all_train):
    """number of contiguous runs per class of `idx`, measured in positions within that class's outer-train list"""
    out = {}
    for c in np.unique(labels[idx]):
        cls_train = np.sort(all_train[labels[all_train] == c])
        pos = np.searchsorted(cls_train, np.sort(idx[labels[idx] == c]))
        out[int(c)] = int(1 + (np.diff(pos) != 1).sum())
    return out


def idx_hash(a):
    return hashlib.sha1(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------- losses
class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__(); self.gamma = gamma; self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = F.softmax(logits, dim=1)[torch.arange(len(targets)), targets]
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            w = w * torch.tensor(self.alpha, device=logits.device)[targets]
        return (w * ce).mean()


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss on the fused embeddings."""
    def __init__(self, temperature):
        super().__init__(); self.t = temperature

    def forward(self, emb, labels):
        b = emb.size(0)
        if b <= 1:
            return torch.tensor(0.0, device=emb.device)
        z = F.normalize(emb, p=2, dim=1)
        sim = z @ z.T / self.t
        eye = torch.eye(b, dtype=torch.bool, device=emb.device)
        pos = (labels[None] == labels[:, None]) & ~eye
        neg = ~pos & ~eye
        valid = pos.any(1) & neg.any(1)
        ninf = torch.finfo(sim.dtype).min
        num = torch.logsumexp(sim.masked_fill(~pos, ninf), 1)
        den = torch.logsumexp(sim.masked_fill(eye, ninf), 1)
        return ((-num + den) * valid).sum() / b


# ---------------------------------------------------------------- model
class AttentionMaskModel(nn.Module):
    """Coarse network: one score in [0, 1] per pixel."""
    def __init__(self):
        super().__init__()
        cin = 3 if Config.DATASET_TYPE == 'wbc' else 1
        self.conv1 = nn.Conv2d(cin, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.intermediate = x
        return torch.sigmoid(self.conv3(x))


class SparsePointEncoder(nn.Module):
    """Embeds each selected pixel (normalized x, y, intensity)."""
    def __init__(self, d):
        super().__init__()
        self.din = 3
        self.mlp = nn.Sequential(nn.Linear(self.din, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Linear(64, d), nn.BatchNorm1d(d))

    def forward(self, pts, image_size):
        b, _, k = pts.shape
        x = pts[:, 0] / (image_size - 1); y = pts[:, 1] / (image_size - 1)
        inp = torch.cat([torch.stack([x, y], -1), pts[:, 2:].transpose(1, 2)], -1).reshape(-1, self.din)
        return self.mlp(inp).view(b, k, -1)


class DynamicInteractionClassifier(nn.Module):
    """SparseAttnNet."""
    def __init__(self):
        super().__init__()
        self.image_size = Config.IMAGE_SIZE
        self.num_heads, self.embed_dim = Config.NUM_HEADS, Config.EMBED_DIM
        self.point_encoder = SparsePointEncoder(self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_proj = nn.Parameter(torch.randn(self.num_heads, self.embed_dim, self.embed_dim) * 0.02)
        self.key_proj = nn.Parameter(torch.randn(self.num_heads, self.embed_dim, self.embed_dim) * 0.02)
        self.attention_mask_model = AttentionMaskModel()
        self.mask_feature_extractor = nn.Sequential(nn.Conv2d(32, Config.MASK_FEATURE_DIM, 1), nn.ReLU(),
                                                    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        if Config.COARSE_ONLY:
            fin = Config.MASK_FEATURE_DIM
        elif Config.FINE_ONLY:
            fin = self.embed_dim
        else:
            fin = self.embed_dim + Config.MASK_FEATURE_DIM
        h = Config.HIDDEN_DIM
        self.fusion_layer = nn.Linear(fin, h // 2)
        self.fc_out = nn.Sequential(nn.Linear(h // 2, h // 4), nn.BatchNorm1d(h // 4), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(h // 4, h // 8), nn.BatchNorm1d(h // 8), nn.ReLU(),
                                    nn.Linear(h // 8, Config.NUM_CLASSES))
        self.residual = nn.Linear(h // 2, Config.NUM_CLASSES)  # not used in forward; kept so that the parameter count and initialization match the reported model

    def fine(self, pts):
        """Linear-complexity fine attention over the k embedded points plus a CLS token."""
        b = pts.shape[0]
        enc = self.point_encoder(pts, self.image_size)
        enc = torch.cat([enc, self.cls_token.expand(b, -1, -1)], 1)
        v = self.value_proj(enc)
        q = torch.einsum('bnd,hde->bhne', enc, self.query_proj)
        kk = torch.einsum('bnd,hde->bhne', enc, self.key_proj)
        kp = torch.relu(kk) + 1e-6
        aw = kp / kp.sum(dim=2, keepdim=True).clamp(min=1e-6)
        ctx = torch.einsum('bhne,bnd->bhed', aw, v)
        ho = torch.einsum('bhne,bhed->bhnd', q, ctx)
        cls = ho[:, :, -1, :].mean(1)
        scores = aw[:, :, :-1, :].mean(3)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0)
        return cls, scores

    def forward(self, sparse_points, k, imgs4d, mask_intermediate=None):
        """imgs4d: [B,C,H,W]. mask_intermediate: coarse conv1 features if already computed."""
        if mask_intermediate is None:
            self.attention_mask_model(imgs4d)
            mask_intermediate = self.attention_mask_model.intermediate
        mf = self.mask_feature_extractor(mask_intermediate)
        if Config.COARSE_ONLY:
            cls, scores, fused_in = None, None, mf
        else:
            cls, scores = self.fine(sparse_points)
            fused_in = cls if Config.FINE_ONLY else torch.cat([cls, mf], -1)
        fused = self.fusion_layer(fused_in)
        return fused, self.fc_out(fused), scores


def to4d(imgs):
    if Config.DATASET_TYPE == 'wbc' or imgs.dim() == 4:
        return imgs
    return imgs.unsqueeze(1)


def select_points(imgs, attention_mask, k, generator=None):
    """Zero a 5-pixel border of the coarse map and take the top-k pixels.
    RANDOM_SELECT (ablation): k uniformly random non-border pixels instead."""
    b = imgs.shape[0]
    h, w = imgs.shape[-2:]
    bd = Config.BORDER
    if Config.RANDOM_SELECT:
        valid = torch.zeros(h, w, dtype=torch.bool, device=imgs.device)
        valid[bd:h - bd, bd:w - bd] = True
        cand = valid.flatten().nonzero().squeeze(1)
        r = torch.rand(b, cand.numel(), device=imgs.device, generator=generator)
        idx = cand[r.argsort(1)[:, :k]]
    else:
        m = attention_mask.clone()
        m[..., :bd, :] = 0; m[..., -bd:, :] = 0; m[..., :, :bd] = 0; m[..., :, -bd:] = 0
        _, idx = m.reshape(b, -1).topk(k, dim=-1)
    flat = imgs.mean(1) if Config.DATASET_TYPE == 'wbc' else imgs.reshape(b, h, w)
    inten = flat.reshape(b, -1).gather(1, idx)
    return torch.stack([idx % w, idx // w, inten], -1).transpose(1, 2).float(), idx


def full_k():
    s = Config.IMAGE_SIZE - 2 * Config.BORDER
    return s * s


# ---------------------------------------------------------------- baselines
def get_model(model_type):
    if model_type == 'ours':
        return DynamicInteractionClassifier()
    import timm
    from torchvision import models
    pre = Config.PRETRAINED
    if model_type == 'resnet50':
        m = models.resnet50(weights='IMAGENET1K_V1' if pre else None)
        m.fc = nn.Linear(m.fc.in_features, Config.NUM_CLASSES); return m
    if model_type == 'mobilenet':
        m = models.mobilenet_v2(weights='IMAGENET1K_V1' if pre else None)
        m.classifier[1] = nn.Linear(m.last_channel, Config.NUM_CLASSES); return m
    name = {'ConvNeXt': 'convnext_tiny', 'DeiT': 'deit_small_patch16_224',
            'ViTSmall': 'vit_small_patch16_224', 'SwinTiny': 'swin_tiny_patch4_window7_224'}[model_type]
    if pre:  # ImageNet weights from the authors' release files, checked by SHA-256 (see fetch_weights.py)
        import fetch_weights
        return timm.create_model(name, pretrained=True, num_classes=Config.NUM_CLASSES,
                                 pretrained_cfg_overlay=dict(file=fetch_weights.path_for(name), hf_hub_id=None, url=None))
    return timm.create_model(name, pretrained=False, num_classes=Config.NUM_CLASSES)


# ---------------------------------------------------------------- metrics
def compute_metrics(y, pred, prob, n_classes):
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    sup = cm.sum(1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        sens = np.where(sup > 0, tp / sup, 0.0)
        spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        f1 = np.where(prec + sens > 0, 2 * prec * sens / (prec + sens), 0.0)
    w = sup / sup.sum()
    conf = prob.max(1); correct = (prob.argmax(1) == y)
    bins = np.linspace(0, 1, 16); ece = 0.0  # expected calibration error, 15 equal-width bins
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    nll = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())
    return {
        'accuracy': float((pred == y).mean()),
        'weighted': {'precision': float((prec * w).sum()), 'sensitivity': float((sens * w).sum()),
                     'specificity': float((spec * w).sum()), 'f1': float((f1 * w).sum())},
        'macro': {'precision': float(prec.mean()), 'sensitivity': float(sens.mean()),
                  'specificity': float(spec.mean()), 'f1': float(f1.mean())},
        'per_class': {'precision': prec.tolist(), 'sensitivity': sens.tolist(),
                      'specificity': spec.tolist(), 'f1': f1.tolist(), 'support': sup.astype(int).tolist()},
        'ece15': float(ece), 'nll': nll, 'confusion_matrix': cm.tolist(),
    }
