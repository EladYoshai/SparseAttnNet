"""Print the result tables from the results folder: classification metrics of every run (mean ± standard
deviation over the five outer test folds), per-class sensitivity and precision of SparseAttnNet, and the
ablation table. Standard deviations are population standard deviations over folds, as in the paper.
Usage: python summarize.py [--results results]
"""
import os, json, glob, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = {'event': ['SW480', 'SW620'], 'holos': ['ACT', 'ACT7d', 'Non-activated'],
           'opd': ['ACT', 'ACT7d', 'Non-activated'], 'wbc': ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']}
MAIN = ['ours', 'resnet50', 'mobilenet', 'ConvNeXt', 'DeiT', 'ViTSmall', 'SwinTiny',
        'ConvNeXt_pretrained', 'DeiT_pretrained', 'ViTSmall_pretrained', 'SwinTiny_pretrained']
ABLATIONS = ['ours', 'fixed_matched', 'fixed_k1500', 'fixed_k3000', 'fixed_k5000', 'fixed_k4000', 'fixed_k8000',
             'fixed_k12000', 'full_image', 'no_distill', 'no_contrast', 'coarse_only', 'fine_only', 'random_matched']


def pm(d, scale=100):
    return f"{d['mean']*scale:6.2f} ± {d['std']*scale:5.2f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); a = ap.parse_args()
    base = os.path.join(ROOT, a.results)
    for ds in ['event', 'holos', 'opd', 'wbc']:
        if not os.path.isdir(os.path.join(base, ds)): continue
        print(f'\n=== {ds}: classification (weighted over classes), % ===')
        print(f"{'model':22s} {'accuracy':>15s} {'precision':>15s} {'sensitivity':>15s} {'specificity':>15s} {'F1':>15s} {'ECE':>15s} {'k':>8s}")
        for tag in MAIN:
            p = os.path.join(base, ds, tag, 'summary.json')
            if not os.path.exists(p): continue
            s = json.load(open(p)); w = s['weighted']
            k = f"{s['k']['mean']:8.0f}" if s.get('k') else ''
            print(f"{tag:22s} {pm(s['accuracy'])} {pm(w['precision'])} {pm(w['sensitivity'])} {pm(w['specificity'])} "
                  f"{pm(w['f1'])} {s['ece15']['mean']:6.3f} ± {s['ece15']['std']:5.3f} {k}")
        fs = sorted(glob.glob(os.path.join(base, ds, 'ours', 'fold_*', 'metrics.json')))
        if fs:
            ms = [json.load(open(f))['per_class'] for f in fs]
            print(f'--- {ds}: SparseAttnNet per class (mean over folds), % ---')
            for i, c in enumerate(CLASSES[ds]):
                sens = np.mean([m['sensitivity'][i] for m in ms]) * 100; prec = np.mean([m['precision'][i] for m in ms]) * 100
                print(f'{c:15s} sensitivity {sens:6.2f}  precision {prec:6.2f}')
        rows = [t for t in ABLATIONS if os.path.exists(os.path.join(base, ds, t, 'summary.json'))]
        if len(rows) > 1:
            st = json.load(open(os.path.join(base, 'paired_stats.json'))) if os.path.exists(os.path.join(base, 'paired_stats.json')) else {}
            print(f'--- {ds}: ablations (accuracy %, paired t-test p versus the full model) ---')
            for t in rows:
                s = json.load(open(os.path.join(base, ds, t, 'summary.json')))
                p = st.get(ds, {}).get(t, {}).get('p_t')
                k = f"k {s['k']['mean']:7.0f}" if s.get('k') else ''
                print(f"{t:15s} {pm(s['accuracy'])}  {k:9s} {'' if p is None else ('p < 0.001' if p < 0.001 else f'p = {p:.3f}')}")


if __name__ == '__main__':
    main()
