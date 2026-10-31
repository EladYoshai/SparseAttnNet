"""Faithfulness (deletion and insertion) test for SparseAttnNet. Uses the trained fold checkpoints (best.pth) of run.py.

For each test image the model selects k pixels (coarse top-k) and assigns each a fine-attention score.

RE-SELECTION variants (input-level perturbation; the whole model, including coarse selection, is re-run):
  del_fine            delete the m% selected pixels with the HIGHEST fine score (most relevant first)
  del_fine_least      delete the m% selected pixels with the LOWEST fine score (least relevant first)
  del_coarse          delete the top m% selected pixels ranked by coarse score
  del_random_selected delete m% random selected pixels (control)
  del_random_anywhere delete the same number of random pixels anywhere inside the border (control)
  "delete" = set to the image background value (median of the 5-px border).
  Because the coarse map is recomputed, the model may re-select other pixels; we log the fraction of the deleted
  pixels that are re-selected (overlap_reselected_<variant>).
FROZEN-SELECTION variants (tests the fine branch itself): the selected indices and the coarse features of the
  ORIGINAL image are kept fixed; only the gathered intensities of the chosen points are replaced by background.
  frozen_fine, frozen_fine_least, frozen_random
INSERTION: start from a background-only image and insert the top m% selected pixels (fine ranking) vs the same
  number of random selected pixels (re-selection), m = 10..100 %.

Deletion fractions m = 0,10,...,90 %. Normalized AUC = trapezoidal area divided by the x-range actually used
(0.9), i.e. the mean accuracy over that range (deletion: lower = more faithful; insertion: higher = more faithful).
Output: results/faithfulness/<dataset>.json and <dataset>.png.
Usage: python deletion_test.py --datasets event holos opd wbc [--folds 0,1,2,3,4] [--device cuda] [--results results]
"""
import os, sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import Config

FRACS = [i / 10 for i in range(10)]          # deletion 0..90 %
INS_FRACS = [(i + 1) / 10 for i in range(10)]  # insertion 10..100 %


def norm_auc(xs, ys):
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def background(imgs, ds):
    b = Config.BORDER
    x = imgs.mean(1) if ds == 'wbc' else (imgs[:, 0] if imgs.dim() == 4 else imgs)
    border = torch.cat([x[:, :b, :].flatten(1), x[:, -b:, :].flatten(1),
                        x[:, :, :b].flatten(1), x[:, :, -b:].flatten(1)], 1)
    return border.median(1).values  # [B]


def set_pixels(imgs, flat_idx, values):
    out = imgs.clone()
    b = imgs.shape[0]
    vals = values[:, None].expand_as(flat_idx).contiguous()
    if out.dim() == 3:
        out.view(b, -1).scatter_(1, flat_idx, vals)
    else:
        for c in range(out.shape[1]):
            t = out[:, c].reshape(b, -1).clone(); t.scatter_(1, flat_idx, vals); out[:, c] = t.view_as(out[:, c])
    return out


def copy_pixels(dst, src, flat_idx):
    out = dst.clone(); b = dst.shape[0]
    if out.dim() == 3:
        out.view(b, -1).scatter_(1, flat_idx, src.reshape(b, -1).gather(1, flat_idx))
    else:
        for c in range(out.shape[1]):
            t = out[:, c].reshape(b, -1).clone()
            t.scatter_(1, flat_idx, src[:, c].reshape(b, -1).gather(1, flat_idx)); out[:, c] = t.view_as(out[:, c])
    return out


@torch.no_grad()
def predict(model, imgs, k):
    x4 = core.to4d(imgs)
    mask = model.attention_mask_model(x4).squeeze(1)
    inter = model.attention_mask_model.intermediate
    pts, idx = core.select_points(imgs, mask, k)
    _, out, scores = model(pts, k, x4, mask_intermediate=inter)
    return out.argmax(1), idx, scores.mean(1), mask, pts, inter, x4


def overlap(idx_new, deleted):
    """fraction of deleted pixels that were re-selected, averaged over the batch"""
    if deleted.shape[1] == 0:
        return 0.0
    k = idx_new.shape[1]
    s = 0.0
    for a, d in zip(idx_new, deleted):
        s += torch.isin(d, a).float().mean().item()
    return s / len(idx_new)


@torch.no_grad()
def run_fold(ds, tag, f, device, gen, results='results', limit=None):
    fd = os.path.join(core.ROOT, results, ds, tag, f'fold_{f}')
    met = json.load(open(os.path.join(fd, 'metrics.json')))
    k = met['best_k']
    Config.update(ds, 'ours')
    model = core.get_model('ours').to(device)
    model.load_state_dict(torch.load(os.path.join(fd, 'best.pth'), map_location=device, weights_only=True))
    model.eval()
    x, labels, _ = core.load_cache(ds, Config.IMAGE_SIZE)
    te = np.load(os.path.join(fd, 'indices.npz'))['test']
    if limit: te = te[:limit]
    del_keys = ['del_fine', 'del_fine_least', 'del_coarse', 'del_random_selected', 'del_random_anywhere',
                'frozen_fine', 'frozen_fine_least', 'frozen_random']
    curves = {key: np.zeros(len(FRACS)) for key in del_keys + ['ins_fine', 'ins_random_selected']}
    ov = {key: np.zeros(len(FRACS)) for key in ['del_fine', 'del_fine_least', 'del_coarse', 'del_random_selected']}
    n = 0; nb = 0
    h = w = Config.IMAGE_SIZE; bd = Config.BORDER
    valid = torch.zeros(h, w, dtype=torch.bool, device=device); valid[bd:h - bd, bd:w - bd] = True
    cand = valid.flatten().nonzero().squeeze(1)
    for s in range(0, len(te), 64):
        j = te[s:s + 64]
        imgs = torch.from_numpy(x[j].astype(np.float32) / 255.0).to(device)
        if ds == 'opd': imgs = imgs.unsqueeze(1)
        y = torch.from_numpy(labels[j]).to(device)
        _, idx, fine, mask, pts, inter, x4 = predict(model, imgs, k)
        coarse = mask.flatten(1).gather(1, idx)
        bg = background(imgs, ds)
        rank_f = fine.argsort(1, descending=True)              # positions within the selected set
        rank_fl = fine.argsort(1, descending=False)
        rank_r = torch.rand(idx.shape, device=device, generator=gen).argsort(1)
        orders = {'del_fine': idx.gather(1, rank_f), 'del_fine_least': idx.gather(1, rank_fl),
                  'del_coarse': idx.gather(1, coarse.argsort(1, descending=True)),
                  'del_random_selected': idx.gather(1, rank_r),
                  'del_random_anywhere': cand[torch.rand(len(j), cand.numel(), device=device, generator=gen).argsort(1)]}
        blank = set_pixels(imgs, cand[None].expand(len(j), -1).contiguous(), bg)
        for i, fr in enumerate(FRACS):
            m = int(round(fr * k))
            for key, order in orders.items():
                dele = order[:, :m].contiguous()
                im = set_pixels(imgs, dele, bg) if m > 0 else imgs
                pr, idx2 = predict(model, im, k)[:2]
                curves[key][i] += (pr == y).sum().item()
                if key in ov: ov[key][i] += overlap(idx2, dele)
            for key, rank in [('frozen_fine', rank_f), ('frozen_fine_least', rank_fl), ('frozen_random', rank_r)]:
                p2 = pts.clone()
                if m > 0:
                    pos = rank[:, :m]
                    p2[:, 2, :] = p2[:, 2, :].scatter(1, pos, bg[:, None].expand_as(pos).contiguous())
                out = model(p2, k, x4, mask_intermediate=inter)[1]
                curves[key][i] += (out.argmax(1) == y).sum().item()
            mi = max(int(round(INS_FRACS[i] * k)), 1)
            for key, order in [('ins_fine', orders['del_fine']), ('ins_random_selected', orders['del_random_selected'])]:
                im = copy_pixels(blank, imgs, order[:, :mi].contiguous())
                curves[key][i] += (predict(model, im, k)[0] == y).sum().item()
        n += len(j); nb += 1
    res = {key: (v / n).tolist() for key, v in curves.items()}
    res.update({f'overlap_reselected_{key}': (v / nb).tolist() for key, v in ov.items()})
    return res, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['event', 'holos'])
    ap.add_argument('--tag', default='ours'); ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--results', default='results')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    device = torch.device(a.device)
    od = os.path.join(core.ROOT, a.results, 'faithfulness'); os.makedirs(od, exist_ok=True)
    for ds in a.datasets:
        gen = torch.Generator(device=device); gen.manual_seed(Config.SEED)
        per_fold = {}
        for f in [int(s) for s in a.folds.split(',')]:
            c, k = run_fold(ds, a.tag, f, device, gen, a.results, a.limit); per_fold[f] = {'k': k, 'curves': c}
            print(ds, f, 'k', k, {kk: round(norm_auc(INS_FRACS if kk.startswith('ins') else FRACS, v), 4)
                                  for kk, v in c.items() if not kk.startswith('overlap')}, flush=True)
        keys = list(per_fold[next(iter(per_fold))]['curves'].keys())
        mean = {kk: np.mean([per_fold[f]['curves'][kk] for f in per_fold], 0).tolist() for kk in keys}
        auc = {}
        for kk in keys:
            if kk.startswith('overlap'): continue
            xs = INS_FRACS if kk.startswith('ins') else FRACS
            v = [norm_auc(xs, per_fold[f]['curves'][kk]) for f in per_fold]
            auc[kk] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        json.dump({'deletion_fractions': FRACS, 'insertion_fractions': INS_FRACS,
                   'auc_definition': 'trapezoidal area / x-range (0.9) = mean accuracy over the fraction range used',
                   'mean_curves': mean, 'normalized_auc': auc, 'per_fold': per_fold},
                  open(os.path.join(od, f'{ds}.json'), 'w'), indent=1)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for kk in [x_ for x_ in keys if x_.startswith('del')]:
            ax[0].plot([100 * f for f in FRACS], np.array(mean[kk]) * 100, marker='o', label=kk[4:])
        for kk in [x_ for x_ in keys if x_.startswith('frozen')]:
            ax[1].plot([100 * f for f in FRACS], np.array(mean[kk]) * 100, marker='o', label=kk)
        for kk in [x_ for x_ in keys if x_.startswith('ins')]:
            ax[2].plot([100 * f for f in INS_FRACS], np.array(mean[kk]) * 100, marker='o', label=kk[4:])
        ax[0].set_title('deletion (re-selection)'); ax[1].set_title('deletion (frozen selection)'); ax[2].set_title('insertion')
        ax[0].set_xlabel('selected pixels removed (%)'); ax[1].set_xlabel('selected pixels removed (%)')
        ax[2].set_xlabel('selected pixels inserted (%)')
        for x_ in ax: x_.set_ylabel('test accuracy (%)'); x_.legend(fontsize=8); x_.grid(alpha=.3)
        fig.suptitle(ds); fig.tight_layout(); fig.savefig(os.path.join(od, f'{ds}.png'), dpi=150)
        print('normalized AUC', ds, json.dumps(auc, indent=1))


if __name__ == '__main__':
    main()
