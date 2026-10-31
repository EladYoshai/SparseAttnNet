"""All paper figures from the results folder. Missing results are skipped where possible.
Output: figures/{accuracy_vs_flops,confusion_matrices,reliability,deletion_curves,ablation_k}.{png,pdf}
Needs: run.py results, results/faithfulness/*.json (deletion_test.py), results/benchmark/flops_by_k.json
(flops_by_k.py) and, optionally, results/benchmark/<dataset>_k*.json (benchmark.py) for the baseline FLOPs.
Usage: python make_figures.py"""
import os, json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
R = os.path.dirname(os.path.abspath(__file__)); F = f'{R}/figures'; os.makedirs(F, exist_ok=True)
for _d in ['/mnt/c/Windows/Fonts', 'C:/Windows/Fonts', '/usr/share/fonts/truetype/msttcorefonts']:
    for _f in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf', 'Times_New_Roman.ttf']:
        if os.path.exists(os.path.join(_d, _f)): font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})

# ---------------------------------------------------------------- accuracy versus FLOPs
def acc(tag, ds):
    p = f'{R}/results/{ds}/{tag}/summary.json'
    return 100 * json.load(open(p))['accuracy']['mean'] if os.path.exists(p) else None
FK = json.load(open(f'{R}/results/benchmark/flops_by_k.json'))
def fk(ds, k):  # FLOPs of SparseAttnNet at budget k (nearest computed)
    d = {int(a): b for a, b in FK[ds].items()}; return d[min(d, key=lambda z: abs(z - k))]
CNN = {'resnet50': ('ResNet50', 1.681, 5.367), 'mobilenet': ('MobileNetV2', 0.131, 0.409)}  # GFLOPs at 135 and 256 px
TR = {'ConvNeXt': ('ConvNeXt-T', 4.470), 'DeiT': ('DeiT-S', 4.608), 'ViTSmall': ('ViT-S', 4.608), 'SwinTiny': ('Swin-T', 4.508)}  # 224 px
for _p in glob.glob(f'{R}/results/benchmark/*_k[0-9]*.json'):  # values measured by benchmark.py replace the defaults above
    _b = json.load(open(_p))
    for _r in _b['rows']:
        if _r['model'] in CNN:
            n_, f135, f256 = CNN[_r['model']]
            CNN[_r['model']] = (n_, f135, _r['GFLOPs_total']) if _b['dataset'] == 'wbc' else (n_, _r['GFLOPs_total'], f256)
        elif _r['model'] in TR:
            TR[_r['model']] = (TR[_r['model']][0], _r['GFLOPs_total'])
ABL = {'event': [('fixed_k1500', 1500), ('fixed_k3000', 3000), ('fixed_k5000', 5000), ('full_image', 125 * 125)],
       'wbc': [('fixed_k4000', 4000), ('fixed_k8000', 8000), ('fixed_k12000', 12000), ('full_image', 246 * 246)]}
BLUE, LBLUE, GRAY, INK, INK2, BG = '#2a78d6', '#9cc3ee', '#8a8985', '#0b0b0b', '#52514e', '#ffffff'
fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.9), facecolor=BG)
for ax, ds, title in zip(axs.flat, ['event', 'holos', 'opd', 'wbc'], ['Event camera', 'Raw holograms', 'OPD', 'WBC']):
    ax.set_facecolor(BG); ys = []; col = []  # col: labels of the high-FLOPs cluster, placed in a column
    def pt(x, y, label, kind, dx=6, dy=-3):
        if y is None: return
        ys.append(y)
        if x > 3 and label:
            col.append((x, y, label)); label = ''
        st = {'ours': dict(s=45, color=BLUE, zorder=5), 'var': dict(s=20, color=LBLUE, zorder=4),
              'base': dict(s=25, color=GRAY, zorder=3), 'pre': dict(s=25, facecolor='none', edgecolor=GRAY, linewidth=1.5, zorder=3)}[kind]
        ax.scatter(x, y, edgecolors=st.pop('edgecolor', BG), linewidths=st.pop('linewidth', 1.5), **st)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6.5,
                    color=INK if kind == 'ours' else INK2, weight='bold' if kind == 'ours' else 'normal')
    for m, (n, f135, f256) in CNN.items(): pt(f256 if ds == 'wbc' else f135, acc(m, ds), n, 'base', dx=(-14 if (ds == 'event' and m == 'resnet50') else 6), dy=(6 if (ds == 'event' and m == 'resnet50') else -3))
    for m, (n, f) in TR.items():
        pt(f, acc(m, ds), n, 'base')
        if ds == 'event': pt(f, acc(m + '_pretrained', ds), n + ' (pretrained)', 'pre')
    curve = [(fk(ds, k), acc(tag, ds), tag, k) for tag, k in ABL.get(ds, []) if acc(tag, ds) is not None]
    if curve:
        curve.sort(); ax.plot([c[0] for c in curve], [c[1] for c in curve], color=LBLUE, lw=1.2, zorder=2)
        for i, (x, y, tag, k) in enumerate(curve):
            lab = 'full image' if tag == 'full_image' else ''  # fixed-k points: values given in the caption
            pt(x, y, lab, 'var', dx=5, dy=-11)
    s = json.load(open(f'{R}/results/{ds}/ours/summary.json'))
    pt(fk(ds, s['k']['mean']), 100 * s['accuracy']['mean'], 'SparseAttnNet', 'ours', dx=(-30 if ds == 'event' else -8), dy=(-15 if ds == 'holos' else 7))
    ax.set_xscale('log'); ax.set_xlim(0.02, 60)
    lo, hi = min(ys), max(ys); y0, y1 = lo - 0.15 * (hi - lo) - 0.3, hi + 0.12 * (hi - lo) + 0.3; ax.set_ylim(y0, y1)
    gap = 0.05 * (y1 - y0); col.sort(key=lambda c: c[1]); pos = []
    for _, y, _ in col:  # spread label heights so that none overlap, keep order
        pos.append(max(y, pos[-1] + gap) if pos else y)
    if pos: pos[-1] = min(pos[-1], y1 - 0.4 * gap)
    for i in range(len(pos) - 2, -1, -1): pos[i] = min(pos[i], pos[i + 1] - gap)
    for (x, y, label), p in zip(col, pos):
        ax.annotate(label, (x, y), xytext=(9.5, p), textcoords='data', fontsize=6.5, color=INK2, va='center',
                    arrowprops=dict(arrowstyle='-', color='#c9c8c3', lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_title(title, loc='left', color=INK)
    ax.grid(True, color='#e6e5e0', linewidth=0.8); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.tick_params(colors=INK2); ax.set_xlabel('GFLOPs per image (log scale)', color=INK2)
    ax.set_ylabel('Accuracy (%)', color=INK2)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=BLUE, ms=8, label='SparseAttnNet (adaptive k)'),
     Line2D([], [], marker='o', ls='', color=LBLUE, ms=6, label='SparseAttnNet, fixed k or full image'),
     Line2D([], [], marker='o', ls='', color=GRAY, ms=6, label='Baselines, trained from scratch'),
     Line2D([], [], marker='o', ls='', mfc='none', mec=GRAY, ms=6, label='Baselines, ImageNet-pretrained')]
fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=7)
plt.tight_layout(rect=(0, 0.07, 1, 1))
for e in ['png', 'pdf']: plt.savefig(f'{R}/figures/accuracy_vs_flops.{e}', dpi=600 if e == 'png' else None, bbox_inches='tight')
plt.close(fig); print('saved accuracy_vs_flops')

# ---------------------------------------------------------------- other figures
DS = ['event', 'holos', 'opd', 'wbc']; TITLE = {'event': 'Event camera', 'holos': 'Raw holograms', 'opd': 'OPD', 'wbc': 'WBC'}
CL = {'event': ['SW480', 'SW620'], 'holos': ['ACT', 'ACT7d', 'Non-act.'], 'opd': ['ACT', 'ACT7d', 'Non-act.'],
      'wbc': ['Baso', 'Eos', 'Lymph', 'Mono', 'Neut']}
BLUE, LBLUE, GRAY, ORANGE, INK, INK2 = '#2a78d6', '#9cc3ee', '#8a8985', '#d9822b', '#0b0b0b', '#52514e'
def preds(ds, tag):
    return [np.load(f'{R}/results/{ds}/{tag}/fold_{f}/test_predictions.npz') for f in range(5)]
def save(fig, name):
    for e in ['png', 'pdf']: fig.savefig(f'{F}/{name}.{e}', dpi=600 if e == 'png' else None, bbox_inches='tight')
    plt.close(fig); print('saved', name)
def style(ax):
    [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(colors=INK2); ax.grid(True, color='#e6e5e0', lw=0.8)

# 1. confusion matrices of SparseAttnNet, summed over the five test folds
fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.2))
for ax, ds in zip(axs.flat, DS):
    n = len(CL[ds]); cm = np.zeros((n, n), int)
    for p in preds(ds, 'ours'):
        for t, q in zip(p['y'], p['pred']): cm[t, q] += 1
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap='Blues', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=7, color='white' if rn[i, j] > 0.6 else INK)
    ax.set_xticks(range(n), CL[ds]); ax.set_yticks(range(n), CL[ds])
    ax.set_xlabel('Predicted class', color=INK2); ax.set_ylabel('True class', color=INK2); ax.set_title(TITLE[ds], loc='left')
    [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); save(fig, 'confusion_matrices')

# 2. reliability diagrams (15 bins) for SparseAttnNet and the convolutional baselines
fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.0))
for ax, ds in zip(axs.flat, DS):
    ax.plot([0, 1], [0, 1], color='#c9c8c3', lw=1, ls='--')
    for tag, lab, col in [('ours', 'SparseAttnNet', BLUE), ('resnet50', 'ResNet50', GRAY), ('mobilenet', 'MobileNetV2', ORANGE)]:
        P = preds(ds, tag); prob = np.concatenate([p['prob'] for p in P]); y = np.concatenate([p['y'] for p in P])
        conf = prob.max(1); corr = (prob.argmax(1) == y).astype(float)
        b = np.minimum((conf * 15).astype(int), 14); xs, ys = [], []
        for i in range(15):
            m = b == i
            if m.sum() >= 10: xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ece = np.mean([json.load(open(f'{R}/results/{ds}/{tag}/fold_{f}/metrics.json'))['ece15'] for f in range(5)])
        ax.plot(xs, ys, marker='o', ms=4, lw=1.5, color=col, label=f'{lab} (ECE {ece:.2f})')
    ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.01); ax.set_xlabel('Mean predicted probability', color=INK2); ax.set_ylabel('Accuracy', color=INK2)
    ax.set_title(TITLE[ds], loc='left'); ax.legend(frameon=False, loc='upper left'); style(ax)
fig.tight_layout(); save(fig, 'reliability')

# 3. deletion curves: attention-ordered versus random removal of the selected pixels
fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.4))
for ax, ds in zip(axs.flat, DS):
    d = json.load(open(f'{R}/results/faithfulness/{ds}.json')); x = np.array(d['deletion_fractions']) * 100
    ax.plot(x, np.array(d['mean_curves']['del_fine']) * 100, color=BLUE, lw=1.8, marker='o', ms=3, label='Highest fine attention first')
    ax.plot(x, np.array(d['mean_curves']['del_random_selected']) * 100, color=GRAY, lw=1.8, marker='o', ms=3, label='Random order')
    ax.set_xlabel('Selected pixels removed (%)', color=INK2); ax.set_ylabel('Accuracy (%)', color=INK2)
    ax.set_title(TITLE[ds], loc='left'); style(ax)
axs[0, 0].legend(frameon=False, loc='lower left'); fig.tight_layout(); save(fig, 'deletion_curves')

# 4. ablation: k trajectories (all datasets, all folds) + component ablation (event, WBC)
fig = plt.figure(figsize=(6.5, 6.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]; ax = axs[0]
for ds, col in zip(DS, [BLUE, ORANGE, GRAY, '#3a9b5c']):
    for f in range(5):
        h = json.load(open(f'{R}/results/{ds}/ours/fold_{f}/history.json'))
        k = np.array(h['k_used'], float); ax.plot(h['epoch'], 100 * k / k[0], color=col, lw=1, alpha=0.8, label=f'{TITLE[ds]} (start {int(k[0]):,})' if f == 0 else None)
ax.set_xlabel('Epoch', color=INK2); ax.set_ylabel('k, % of its starting value', color=INK2)
ax.set_title('(a) Pixel budget during training, five folds per dataset', loc='left'); ax.legend(frameon=False, ncol=2); style(ax)
VAR = [('ours', 'Full model'), ('fixed_matched', 'Fixed k = selected k'), ('full_image', 'Full image'), ('no_distill', 'No distillation'),
       ('no_contrast', 'No contrastive loss'), ('coarse_only', 'No fine attention'), ('fine_only', 'No feature fusion'), ('random_matched', 'Random pixels (trained)')]
for ax, ds in zip(axs[1:], ['event', 'wbc']):
    m, s = [], []
    for tag, _ in VAR:
        a = [json.load(open(f'{R}/results/{ds}/{tag}/fold_{f}/metrics.json'))['accuracy'] * 100 for f in range(5)]
        m.append(np.mean(a)); s.append(np.std(a))
    yy = np.arange(len(VAR))[::-1]
    for i, (y_, v, e) in enumerate(zip(yy, m, s)):
        ax.errorbar(v, y_, xerr=e, fmt='o', ms=7 if i == 0 else 5, color=BLUE if i == 0 else GRAY, ecolor='#c9c8c3', elinewidth=1.2, capsize=2)
        ax.text(v, y_ + 0.3, f'{v:.2f}', fontsize=6.5, color=INK2, ha='center')
    ax.axvline(m[0], color=BLUE, lw=0.8, ls=':'); ax.set_yticks(yy, [v[1] for v in VAR]); ax.set_ylim(-0.6, len(VAR) - 0.3)
    ax.set_xlabel('Accuracy (%)', color=INK2); ax.set_title(f'({"b" if ds == "event" else "c"}) Ablation, {TITLE[ds]}', loc='left'); style(ax); ax.grid(axis='y', visible=False)
    if ds == 'wbc': ax.set_yticklabels([])
fig.tight_layout(); save(fig, 'ablation_k')
