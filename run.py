"""Nested 5-fold cross-validation of one (dataset, model, variant).

For each outer fold: 20 % of the outer-training images form an inner validation set. Every model trains for
exactly 200 epochs. The checkpoint with the best inner-validation accuracy (ties: lower inner-validation loss)
is kept, and the outer test fold is evaluated once, with that checkpoint (and, for SparseAttnNet, the k it used).

Usage:
  python run.py --dataset event --model ours --fast
  python run.py --dataset event --model resnet50 --fast
  python run.py --dataset event --model DeiT --pretrained --fast --tag DeiT_pretrained
Ablations (model 'ours'): --fixed_k N | --full_image | --k_from ours (k fixed per fold to the k selected by the
full model) | --no_distill | --no_contrast | --random_select | --coarse_only | --fine_only
Output: results/<dataset>/<tag>/summary.json and fold_<i>/{metrics.json, test_predictions.npz, history.json, ...}
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import Config

TRAIN_SEED = Config.SEED


class TestGuard:
    """Raises if the outer test fold is evaluated before training finished, or more than once."""
    def __init__(self):
        self.calls = 0; self.training_done = False

    def check(self):
        if not self.training_done:
            raise RuntimeError('test fold touched before training finished')
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError('test fold evaluated more than once')


class GpuLoader:
    """--fast path: the uint8 image cache stays on the GPU and batches are built by indexing on the GPU.
    It gives the same batches, in the same order, with the same values, as the torch DataLoader path."""
    class _Idx:
        def __init__(self, idx): self.idx = np.asarray(idx)

    def __init__(self, xg, labels_g, idx, ds, mt, batch_size, shuffle, generator, lut):
        self.xg, self.yg, self.ds, self.mt = xg, labels_g, ds, mt
        self.idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        self.dataset = GpuLoader._Idx(idx)
        self.bs, self.shuffle, self.g, self.lut = batch_size, shuffle, generator, lut

    def __len__(self):
        return (len(self.idx_t) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.idx_t)
        if self.shuffle:  # same random draws as torch.utils.data.DataLoader with RandomSampler
            torch.empty((), dtype=torch.int64).random_(generator=self.g)
            order = self.idx_t[torch.randperm(n, generator=self.g)]
            torch.randperm(n, generator=self.g)
        else:
            torch.empty((), dtype=torch.int64).random_()
            order = self.idx_t
        order = order.to(self.xg.device)
        for s in range(0, n, self.bs):
            j = order[s:s + self.bs]
            raw = self.xg[j].long()
            if self.mt != 'ours':
                if raw.dim() == 3:
                    raw = raw.unsqueeze(1).expand(-1, 3, -1, -1)
                t = torch.stack([self.lut[c][raw[:, c]] for c in range(3)], 1) if self.lut.dim() == 2 else self.lut[raw]
            else:
                t = self.lut[raw]
                if self.ds == 'opd':
                    t = t.unsqueeze(1)
            yield t.contiguous(), self.yg[j]


def make_lut(device, normalize):
    base = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0)
    if not normalize:
        return base.to(device)
    return torch.stack([(base - core.IMAGENET_MEAN.view(-1)[c]) / core.IMAGENET_STD.view(-1)[c] for c in range(3)]).to(device)


def forward_ours(model, imgs, k, train, gen=None):
    x4 = core.to4d(imgs)
    mask = model.attention_mask_model(x4).squeeze(1)
    if Config.COARSE_ONLY:
        pts = None
    else:
        pts, _ = core.select_points(imgs, mask, k, generator=gen)
    # training: the coarse network runs again inside forward (as in the original implementation, which keeps the
    # BatchNorm running statistics identical); evaluation: its features are reused (identical output)
    inter = None if train else model.attention_mask_model.intermediate
    emb, out, scores = model(pts, k, x4, mask_intermediate=inter)
    return emb, out, scores, mask, pts


def evaluate(model, loader, k, crit, device, gen_seed=0):
    model.eval()
    ys, ps, ls = [], [], []
    gen = torch.Generator(device=device); gen.manual_seed(gen_seed)
    with torch.no_grad():
        for imgs, y in loader:
            imgs, y = imgs.to(device), y.to(device)
            if Config.MODEL_TYPE == 'ours':
                _, out, _, _, _ = forward_ours(model, imgs, k, train=False, gen=gen)
            else:
                out = model(imgs)
            ls.append(crit(out, y).item() * len(y))
            ys.append(y.cpu().numpy()); ps.append(F.softmax(out, 1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p, float(np.sum(ls) / len(y))


def train_fold(model, tr_loader, va_loader, device, crit, fold_dir, log):
    ours = Config.MODEL_TYPE == 'ours'
    adaptive = ours and Config.FIXED_K is None and not Config.FULL_IMAGE
    if not ours:
        k = None
    elif Config.FULL_IMAGE:
        k = core.full_k()
    elif Config.FIXED_K is not None:
        k = Config.FIXED_K
    else:
        k = Config.INITIAL_K
    use_distill = ours and Config.DISTILLATION_WEIGHT > 0 and not Config.COARSE_ONLY
    contr = core.ContrastiveLoss(Config.CONTRASTIVE_TEMPERATURE) if ours and Config.CONTRASTIVE_WEIGHT > 0 else None

    if ours:
        if Config.COARSE_ONLY:
            # no fine branch, hence no distillation signal: the coarse network is trained by the classifier loss
            cls_params = list(model.parameters()); mask_opt = None
        else:
            cls_params = [p for n, p in model.named_parameters() if 'attention_mask_model' not in n]
            mask_opt = optim.AdamW(model.attention_mask_model.parameters(), lr=Config.MASK_LEARNING_RATE)
        opt = optim.AdamW(cls_params, lr=Config.LEARNING_RATE, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=5, min_lr=1e-6)
        msched = (optim.lr_scheduler.ReduceLROnPlateau(mask_opt, mode='min', factor=0.9, patience=5, min_lr=1e-6)
                  if mask_opt else None)
    else:
        opt = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=5, min_lr=1e-6)
        mask_opt = msched = None

    hist = {'epoch': [], 'k_used': [], 'train_loss': [], 'distill_loss': [], 'contrast_loss': [],
            'val_loss': [], 'val_acc': [], 'lr': [], 'ema': [], 'trend': []}
    best = (-1.0, float('inf')); best_info = None
    ema = prev_ema = None
    gen = torch.Generator(device=device); gen.manual_seed(TRAIN_SEED)
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        tl = dl = cl = 0.0; nb = 0
        for imgs, y in tr_loader:
            imgs, y = imgs.to(device), y.to(device)
            if ours:
                emb, out, scores, mask, pts = forward_ours(model, imgs, k, train=True, gen=gen)
                loss = crit(out, y)
                if contr is not None:
                    c = contr(emb, y); loss = loss + Config.CONTRASTIVE_WEIGHT * c; cl += c.item()
                opt.zero_grad()
                loss.backward(retain_graph=use_distill)
                opt.step()
                if use_distill and epoch >= Config.DISTILLATION_WARMUP_EPOCHS:
                    # distillation: the coarse map learns to match the fine-attention distribution
                    mask_opt.zero_grad()
                    b = imgs.shape[0]; h, w = imgs.shape[-2:]
                    ind = (pts[:, 1, :] * w + pts[:, 0, :]).long()
                    na = F.softmax(scores.sum(1), 1)
                    am = torch.zeros(b, h * w, device=device).scatter_(1, ind, na).view(b, h, w)
                    em = am.detach().pow(Config.EMPHASIS_FACTOR)
                    em = em / (em.flatten(1).max(1)[0][:, None, None] + 1e-8)
                    nm = F.softmax(mask.flatten(1), 1).view_as(mask)
                    ne = F.softmax(em.flatten(1), 1).view_as(em)
                    d = F.kl_div(nm.log(), ne.detach(), reduction='batchmean')
                    (Config.DISTILLATION_WEIGHT * d).backward()
                    mask_opt.step(); dl += d.item()
            else:
                out = model(imgs); loss = crit(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        tl /= nb
        yv, pv, vloss = evaluate(model, va_loader, k, crit, device)  # inner validation only
        vacc = float((pv.argmax(1) == yv).mean())
        k_used = k
        # adaptive k: driven by the smoothed training loss
        if adaptive:
            if epoch == 0:
                ema = prev_ema = tl
            else:
                prev_ema = ema; ema = Config.EMA_BETA * ema + (1 - Config.EMA_BETA) * tl
            trend = ema - prev_ema
            pk = k
            k = int(np.clip(k + Config.LR_K if trend >= 0 else k - Config.LR_K, Config.K_MIN, Config.K_MAX))
            k = int(Config.K_MOMENTUM * pk + (1 - Config.K_MOMENTUM) * k)
        else:
            trend = 0.0
        sched.step(vloss)
        if msched: msched.step(vloss)
        for key, val in zip(hist, [epoch + 1, k_used, tl, dl / max(nb, 1), cl / max(nb, 1), vloss, vacc,
                                   opt.param_groups[0]['lr'], ema, trend]):
            hist[key].append(val)
        improved = (vacc > best[0]) or (vacc == best[0] and vloss < best[1])
        if improved:
            best = (vacc, vloss)
            best_info = {'state': {kk: v.detach().clone() for kk, v in model.state_dict().items()},
                         'epoch': epoch + 1, 'k': k_used, 'val_acc': vacc, 'val_loss': vloss}
        log(f'ep {epoch+1:3d} loss {tl:.4f} distill {dl/max(nb,1):.4f} val_loss {vloss:.4f} val_acc {vacc:.4f} '
            f'k_used {k_used} next_k {k} lr {opt.param_groups[0]["lr"]:.2e}{" *" if improved else ""}')
    json.dump(hist, open(os.path.join(fold_dir, 'history.json'), 'w'))
    if ours:  # checkpoint kept for the faithfulness test
        torch.save(best_info['state'], os.path.join(fold_dir, 'best.pth'))
    return best_info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(core.CLASS_DIRS))
    ap.add_argument('--model', required=True, choices=['ours', 'resnet50', 'mobilenet'] + core.TRANSFORMERS)
    ap.add_argument('--fixed_k', type=int); ap.add_argument('--full_image', action='store_true')
    ap.add_argument('--k_from', help='fix k per fold to the k selected by results/<dataset>/<K_FROM> (matched-budget ablations)')
    ap.add_argument('--no_distill', action='store_true'); ap.add_argument('--no_contrast', action='store_true')
    ap.add_argument('--random_select', action='store_true'); ap.add_argument('--coarse_only', action='store_true')
    ap.add_argument('--fine_only', action='store_true'); ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--max_epochs', type=int)
    ap.add_argument('--tag'); ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fast', action='store_true', help='GPU-resident data + cudnn.benchmark (+channels_last for ConvNeXt); same math and batch order')
    a = ap.parse_args(argv)

    Config.update(a.dataset, a.model)
    Config.FIXED_K = a.fixed_k; Config.FULL_IMAGE = a.full_image; Config.RANDOM_SELECT = a.random_select
    Config.COARSE_ONLY = a.coarse_only; Config.FINE_ONLY = a.fine_only; Config.PRETRAINED = a.pretrained
    if a.no_distill: Config.DISTILLATION_WEIGHT = 0.0
    if a.no_contrast: Config.CONTRASTIVE_WEIGHT = 0.0
    if a.max_epochs: Config.MAX_EPOCHS = a.max_epochs
    if a.dataset == 'wbc':  # independent single-cell images: shuffled outer folds + random stratified inner split
        Config.INNER_SPLIT = 'random'
    tag = a.tag or (a.model + ('_pretrained' if a.pretrained else ''))
    run_dir = os.path.join(core.ROOT, a.out, a.dataset, tag); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, 'train.log'), 'a')

    def log(s):
        line = f'[{time.strftime("%H:%M:%S")}] {s}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()

    device = torch.device(a.device)
    cuda = device.type == 'cuda'
    x, labels, paths = core.load_cache(a.dataset, Config.IMAGE_SIZE)
    folds = core.outer_folds(labels, shuffle=(a.dataset == 'wbc'))
    cfg = {k: v for k, v in vars(Config).items() if k.isupper()}
    log(f'RUN {a.dataset}/{tag} args={vars(a)}')
    log(f'config={json.dumps(cfg)}')
    log(f'n={len(labels)} class_counts={np.bincount(labels).tolist()} data_shape={x.shape}')
    fold_results = []
    t0 = time.time()
    if a.fast:
        torch.backends.cudnn.benchmark = True
        xg = torch.from_numpy(x).to(device); yg = torch.from_numpy(labels).to(device)
    for f in [int(s) for s in a.folds.split(',')]:
        tr_out, te = folds[f]
        tr, va = core.inner_split(tr_out, labels)
        # no overlap between train, inner validation and test
        assert not (set(te) & set(tr)) and not (set(te) & set(va)) and not (set(tr) & set(va))
        assert len(tr) + len(va) + len(te) == len(labels)
        runs = core.contiguous_runs(va, labels, tr_out)
        if Config.INNER_SPLIT == 'contiguous':
            assert all(v <= 1 for v in runs.values()), f'inner-val not one contiguous run per class: {runs}'
        fold_dir = os.path.join(run_dir, f'fold_{f}'); os.makedirs(fold_dir, exist_ok=True)
        split_info = {'test_hash': core.idx_hash(te), 'train_hash': core.idx_hash(tr), 'val_hash': core.idx_hash(va),
                      'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                      'train_counts': np.bincount(labels[tr], minlength=Config.NUM_CLASSES).tolist(),
                      'val_counts': np.bincount(labels[va], minlength=Config.NUM_CLASSES).tolist(),
                      'test_counts': np.bincount(labels[te], minlength=Config.NUM_CLASSES).tolist(),
                      'inner_split': Config.INNER_SPLIT}
        json.dump(split_info, open(os.path.join(fold_dir, 'split.json'), 'w'))
        np.savez(os.path.join(fold_dir, 'indices.npz'), train=tr, val=va, test=te)
        log(f'fold {f}: {split_info}')
        if a.k_from:
            Config.FIXED_K = int(json.load(open(os.path.join(core.ROOT, a.out, a.dataset, a.k_from, f'fold_{f}', 'metrics.json')))['best_k'])
            log(f'fold {f}: k fixed to {Config.FIXED_K} (k selected by {a.k_from})')
        core.set_seed(TRAIN_SEED + f)
        g = torch.Generator(); g.manual_seed(TRAIN_SEED + f)
        if a.fast:
            lut = make_lut(device, Config.PRETRAINED and a.model != 'ours')
            mkg = lambda idx, sh: GpuLoader(xg, yg, idx, a.dataset, a.model, Config.BATCH_SIZE, sh, g if sh else None, lut)
            tr_loader, va_loader, te_loader = mkg(tr, True), mkg(va, False), mkg(te, False)
        else:
            mk = lambda idx, sh: DataLoader(core.CachedDataset(x, labels, idx, a.dataset, a.model),
                                            batch_size=Config.BATCH_SIZE, shuffle=sh, num_workers=4,
                                            generator=g if sh else None, persistent_workers=False)
            tr_loader, va_loader, te_loader = mk(tr, True), mk(va, False), mk(te, False)
        model = core.get_model(a.model).to(device)
        if a.fast and a.model == 'ConvNeXt':
            model = model.to(memory_format=torch.channels_last)
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f'params={nparam} ({nparam/1e6:.4f} M)')
        crit = core.FocalLoss(Config.FOCAL_LOSS_GAMMA, Config.FOCAL_LOSS_ALPHA)
        guard = TestGuard()
        tf = time.time()
        if cuda: torch.cuda.reset_peak_memory_stats()
        info = train_fold(model, tr_loader, va_loader, device, crit, fold_dir, log)
        guard.training_done = True
        train_s = time.time() - tf
        model.load_state_dict(info['state'])
        guard.check()
        yt, pt, tloss = evaluate(model, te_loader, info['k'], crit, device)  # the only evaluation on the test fold
        m = core.compute_metrics(yt, pt.argmax(1), pt, Config.NUM_CLASSES)
        m.update({'fold': f, 'best_epoch': info['epoch'], 'best_k': info['k'],
                  'inner_val_acc': info['val_acc'], 'inner_val_loss': info['val_loss'], 'test_loss': tloss,
                  'epochs_trained': Config.MAX_EPOCHS, 'params': nparam, 'train_seconds': train_s,
                  'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0})
        np.savez(os.path.join(fold_dir, 'test_predictions.npz'), y=yt, pred=pt.argmax(1), prob=pt, test_idx=te)
        json.dump(m, open(os.path.join(fold_dir, 'metrics.json'), 'w'), indent=1)
        log(f'fold {f} TEST acc {m["accuracy"]:.4f} macroF1 {m["macro"]["f1"]:.4f} ece {m["ece15"]:.4f} '
            f'epoch {info["epoch"]} k {info["k"]} ({train_s/60:.1f} min)')
        fold_results.append(m)
        del info, model
        if cuda: torch.cuda.empty_cache()

    def ms(vals):
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    fr = fold_results
    s = {'dataset': a.dataset, 'tag': tag, 'args': vars(a), 'n_folds': len(fr),
         'accuracy': ms([r['accuracy'] for r in fr]), 'ece15': ms([r['ece15'] for r in fr]),
         'k': ms([r['best_k'] for r in fr]) if a.model == 'ours' and not a.coarse_only else None,
         'best_epoch': [r['best_epoch'] for r in fr], 'params': fr[0]['params'], 'minutes': (time.time() - t0) / 60}
    for avg in ['weighted', 'macro']:
        s[avg] = {mm: ms([r[avg][mm] for r in fr]) for mm in ['precision', 'sensitivity', 'specificity', 'f1']}
    s['per_fold_accuracy'] = [r['accuracy'] for r in fr]
    json.dump(s, open(os.path.join(run_dir, 'summary.json'), 'w'), indent=1)
    log(f'SUMMARY {a.dataset}/{tag}: acc {s["accuracy"]["mean"]*100:.2f} ± {s["accuracy"]["std"]*100:.2f} '
        f'({s["minutes"]:.1f} min)')
    return s


if __name__ == '__main__':
    main()
